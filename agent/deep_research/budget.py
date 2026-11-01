"""Per-researcher token budgets (second experimental factor, see config.BUDGET_MODES).

native  Claude task budget. Bedrock supports it on InvokeModel / the Messages API but not on Converse,
        so native researchers use Strands' AnthropicModel with the Anthropic SDK's Bedrock client
        (InvokeModel, SigV4 with the runtime role; data stays in Bedrock) instead of BedrockModel.
client  The same budget, counted here and shown to the model on every tool result.

Both modes count the same thing the API counts: tokens the model *sees* in the researcher's loop
(its own output + new tool results), not the history re-sent on every request. That is measured as
    context of the latest call  -  context of the first call  +  output of the latest call
from the usage metadata Strands stores on each assistant message.

The same counter is recorded in both modes (and logged in summary.json -> researcher_budgets),
so the eval can compare how well each mode keeps to the budget.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

import anthropic
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import AfterInvocationEvent, AfterToolCallEvent, BeforeModelCallEvent, BeforeToolCallEvent
from strands.models import CacheConfig
from strands.models.anthropic import AnthropicModel

from .config import BUDGET_OVERRUN_FACTOR, TASK_BUDGET_BETA, Settings

log = logging.getLogger("deep_research.budget")

WRAP_UP_SHARE = 0.70   # client mode: ask to wrap up
STOP_SHARE = 0.90      # client mode: ask to stop calling tools and write findings

_CJK = re.compile(r"[　-鿿가-힯＀-￯]")


def estimate_tokens(text: str) -> int:
    """Rough token estimate for text not yet seen by the model: CJK ~1 token/char, other ~4 chars/token."""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    return cjk + max(len(text) - cjk, 0) // 4


def _context_tokens(usage: dict) -> int:
    return sum(int(usage.get(k, 0) or 0) for k in ("inputTokens", "cacheReadInputTokens", "cacheWriteInputTokens"))


def counted_tokens(messages: list[dict], pending_text: str = "") -> int:
    """Tokens the model has seen in this loop so far (+ an estimate for text it is about to see)."""
    usages = [m["metadata"]["usage"] for m in messages
              if m.get("role") == "assistant" and (m.get("metadata") or {}).get("usage")]
    if not usages:
        return estimate_tokens(pending_text)
    first, last = usages[0], usages[-1]
    seen = _context_tokens(last) - _context_tokens(first) + int(last.get("outputTokens", 0) or 0)
    # tool results appended after the last assistant message are not in any usage yet
    tail_text = ""
    if messages and messages[-1].get("role") == "user":
        for b in messages[-1].get("content", []):
            for c in (b.get("toolResult") or {}).get("content", []) or []:
                tail_text += c.get("text", "") if isinstance(c, dict) else ""
    return max(seen, 0) + estimate_tokens(tail_text) + estimate_tokens(pending_text)


def native_budget_model(settings: Settings, model_id: str, max_tokens: int, budget_tokens: int) -> AnthropicModel:
    """Researcher model with a Claude task budget, served by Bedrock InvokeModel."""
    model = AnthropicModel(
        client_args={"api_key": "unused-bedrock-sigv4"},   # replaced below by the Bedrock client
        model_id=model_id,
        max_tokens=max_tokens,
        use_native_token_count=False,                      # count_tokens is not available on Bedrock
        cache_config=CacheConfig(strategy="auto") if settings.prompt_cache else None,
        params={
            # the Bedrock client moves this header into the body as anthropic_beta
            "extra_headers": {"anthropic-beta": TASK_BUDGET_BETA},
            # set once per researcher and never changed, so it does not break the prompt cache
            "extra_body": {"output_config": {"task_budget": {"type": "tokens", "total": int(budget_tokens)}}},
        },
    )
    if settings.native_anthropic_version:
        model.config["params"]["extra_body"]["anthropic_version"] = settings.native_anthropic_version
    model.client = anthropic.AsyncAnthropicBedrock(aws_region=settings.region, max_retries=6, timeout=900)
    return model


class TokenBudget(HookProvider):
    """Tracks the researcher's counted tokens; in client mode also shows the countdown and enforces it."""

    def __init__(self, ctx: Any, agent_id: str, budget_mode: str, budget_tokens: Optional[int]):
        # budget_mode "none": only measures counted tokens (no budget, nothing shown or enforced)
        self.ctx, self.agent_id, self.mode = ctx, agent_id, budget_mode
        self.total = int(budget_tokens) if budget_tokens else None
        self.overrun_at = self.total * BUDGET_OVERRUN_FACTOR[budget_mode] if self.total else float("inf")
        self.peak = 0
        self.stage_seen: set[str] = set()

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self._before_model)
        registry.add_callback(AfterInvocationEvent, self._after_invocation)
        if self.mode == "client":
            registry.add_callback(BeforeToolCallEvent, self._before_tool)
            registry.add_callback(AfterToolCallEvent, self._after_tool)

    # -------------------------------------------------------------- helpers
    def _used(self, agent: Any, pending: str = "") -> int:
        n = counted_tokens(agent.messages, pending)
        self.peak = max(self.peak, n)
        return n

    def _mark(self, stage: str, used: int) -> None:
        if stage not in self.stage_seen:
            self.stage_seen.add(stage)
            self.ctx.log_event(kind="token_budget", agent=self.agent_id, budget_mode=self.mode, stage=stage,
                               used=used, total=self.total)

    def footer(self, used: int) -> str:
        left = max(self.total - used, 0)
        share = used / self.total
        line = f"[token budget: ~{used:,} of {self.total:,} used ({share:.0%}), ~{left:,} left]"
        if share >= STOP_SHARE:
            self._mark("stop", used)
            return line + (" Budget almost exhausted: do not call more tools. Write your findings now with the "
                           "sources you already have.")
        if share >= WRAP_UP_SHARE:
            self._mark("wrap_up", used)
            return line + (" Start wrapping up: make at most 1-2 more targeted tool calls for the most important "
                           "remaining gap, then write your findings.")
        return line

    # -------------------------------------------------------------- hooks
    def _before_model(self, event: BeforeModelCallEvent) -> None:
        used = self._used(event.agent)
        if used >= self.overrun_at:
            self._mark("overrun_cancel", used)
            event.cancel = f"Stopped: token budget exceeded (~{used:,} of {self.total:,})."

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        used = self._used(event.agent)
        if used >= self.total:
            self._mark("exhausted", used)
            event.cancel_tool = (f"TOKEN_BUDGET_EXHAUSTED: ~{used:,} of {self.total:,} tokens used. Do not call "
                                 "more tools; write your findings now with the sources you already have.")

    def _after_tool(self, event: AfterToolCallEvent) -> None:
        result = event.result or {}
        content = list(result.get("content") or [])
        text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        used = self._used(event.agent, pending=text)
        content.append({"text": "\n" + self.footer(used)})
        event.result = {**result, "content": content}

    def _after_invocation(self, event: AfterInvocationEvent) -> None:
        used = self._used(event.agent)
        self.ctx.record_researcher_budget(self.agent_id, {
            "budget_mode": self.mode, "total": self.total, "counted": used,
            "ratio": round(used / self.total, 3) if self.total else None, "stages": sorted(self.stage_seen)})
