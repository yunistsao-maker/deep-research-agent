"""Deep research pipeline: brief -> supervisor (parallel researchers) -> compression -> report.

Usage:
    from deep_research.pipeline import run_research
    summary = run_research(RunRequest(topic="...", mode="S3"))
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import threading
import time
import traceback
from typing import Any, Optional

from strands import Agent, tool
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import AfterInvocationEvent, BeforeModelCallEvent
from strands.models import CacheConfig
from strands.models.bedrock import BedrockModel
from botocore.config import Config as BotoConfig

from . import prompts
from .budget import TokenBudget, native_budget_model
from .config import MODES, RunRequest, Settings
from .run_context import ResultStore, RunContext, domain_of
from .tools.browser import BrowserSession, make_browser_tools
from .tools.fetch import make_fetch_tool
from .tools.search import make_web_search_tool

log = logging.getLogger("deep_research")

_BOTO_CFG = BotoConfig(read_timeout=900, connect_timeout=30, retries={"max_attempts": 8, "mode": "adaptive"})


def _model(settings: Settings, model_id: str, max_tokens: Optional[int] = None) -> BedrockModel:
    kwargs: dict[str, Any] = {}
    if settings.prompt_cache:
        # Bedrock prompt caching: Strands places cache points on the system prompt, tool specs and the
        # latest message, so each agent-loop turn re-reads the growing conversation at the cache-read rate
        # instead of the full input rate. Applies to every stage and every mode, so the comparison stays fair.
        kwargs["cache_config"] = CacheConfig(strategy="auto")
    return BedrockModel(model_id=model_id, region_name=settings.region, max_tokens=max_tokens or settings.max_tokens,
                        boto_client_config=_BOTO_CFG, **kwargs)


class CostMeter(HookProvider):
    """Keeps the run's token usage live (before each model call) and enforces the hard cap.

    enforce=True (researchers, supervisor): once the run reaches its cost/time cap, the next model call
    is cancelled with a short message, so the agent ends its turn instead of raising. Brief, compression
    and the final report are never cancelled, so a capped run still produces a report.
    """

    def __init__(self, ctx: RunContext, stage: str, enforce: bool = False):
        self.ctx, self.stage, self.enforce = ctx, stage, enforce

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self._before_model)
        registry.add_callback(AfterInvocationEvent, self._after_invocation)

    def _before_model(self, event: BeforeModelCallEvent) -> None:
        self.ctx.sync_usage(self.stage, event.agent)
        if self.enforce:
            state, reason = self.ctx.limit_state()
            if state == "hard":
                self.ctx.log_event(kind="model_call_cancelled", stage=self.stage, agent=event.agent.name,
                                   reason=reason)
                event.cancel = f"Stopped: run {reason}."

    def _after_invocation(self, event: AfterInvocationEvent) -> None:
        self.ctx.sync_usage(self.stage, event.agent, final=True)


def _native_model(settings: Settings, model_id: str, budget_tokens: int):
    """Researcher model for budget_mode 'native' (Bedrock InvokeModel with a Claude task budget)."""
    return native_budget_model(settings, model_id, settings.max_tokens, budget_tokens)


def _text_of(result: Any) -> str:
    try:
        return "".join(b.get("text", "") for b in result.message["content"] if isinstance(b, dict)).strip()
    except Exception:
        return str(result).strip()


def _transcript(messages: list[dict], max_chars: int = 150_000) -> str:
    parts: list[str] = []
    for m in messages:
        for b in m.get("content", []):
            if "text" in b and b["text"].strip():
                parts.append(f"[{m['role']}] {b['text'].strip()}")
            elif "toolUse" in b:
                parts.append(f"[tool call] {b['toolUse'].get('name')}({b['toolUse'].get('input')})")
            elif "toolResult" in b:
                txt = "".join(c.get("text", "") for c in b["toolResult"].get("content", []) if isinstance(c, dict))
                parts.append(f"[tool result]\n{txt}")
    out = "\n\n".join(parts)
    if len(out) > max_chars:  # keep head (plan) and tail (latest evidence + notes)
        out = out[: max_chars // 3] + "\n\n...[middle of transcript omitted]...\n\n" + out[-(2 * max_chars) // 3:]
    return out


class Researcher:
    """One researcher sub-agent with the tool set of the run's mode."""

    def __init__(self, ctx: RunContext, rid: str, subtopic: str):
        self.ctx, self.rid, self.subtopic = ctx, rid, subtopic
        self.browser: Optional[BrowserSession] = None

    def _tools(self) -> list:
        mode = MODES[self.ctx.req.mode]
        tools = [make_web_search_tool(self.ctx, self.rid)]
        if "fetch_url" in mode["tools"]:
            tools.append(make_fetch_tool(self.ctx, self.rid, mode["fetch_scope"]))
        if any(t.startswith("browser_") for t in mode["tools"]):
            s = self.ctx.settings
            self.browser = BrowserSession(s.region, s.browser_identifier)
            tools += make_browser_tools(self.ctx, self.rid, self.browser)
        return tools

    def run(self) -> str:
        ctx, s = self.ctx, self.ctx.settings
        t0 = time.time()
        ctx.log_event(kind="researcher_start", agent=self.rid, subtopic=self.subtopic)
        try:
            req = ctx.req
            budget = req.token_budget                      # None in budget_mode "none"
            if req.budget_mode == "native":
                model = _native_model(s, s.model_id, budget)
            else:
                model = _model(s, s.model_id)
            if budget:
                rule = prompts.BUDGET_RULE["token"].format(token_budget=budget)
                kickoff = f"Research the subtopic now. Token budget: about {budget:,} tokens."
            else:
                rule = prompts.BUDGET_RULE["none"].format(max_tool_calls=req.tool_call_limit)
                kickoff = f"Research the subtopic now. Budget: {req.tool_call_limit} tool calls."
            agent = Agent(
                model=model,
                tools=self._tools(),
                system_prompt=prompts.RESEARCHER_PROMPT.format(
                    today=dt.date.today().isoformat(), subtopic=self.subtopic,
                    tool_guidance=prompts.TOOL_GUIDANCE[req.mode], budget_rule=rule),
                callback_handler=None,
                name=self.rid,
                hooks=[CostMeter(ctx, "researcher", enforce=True),
                       TokenBudget(ctx, self.rid, req.budget_mode, budget)],
            )
            agent(kickoff)
            transcript = _transcript(agent.messages)
        finally:
            if self.browser:
                secs = self.browser.close()
                ctx.log_event(kind="browser_session", agent=self.rid, session_seconds=secs)
        comp = Agent(model=_model(s, s.compress_model_id), callback_handler=None, name=f"{self.rid}-compress",
                     hooks=[CostMeter(ctx, "compress")])
        cres = comp(prompts.COMPRESS_PROMPT.format(subtopic=self.subtopic, transcript=transcript))
        findings = _text_of(cres)
        ctx.log_event(kind="researcher_end", agent=self.rid, elapsed=round(time.time() - t0, 1),
                      tool_calls=ctx.tool_calls_by_agent.get(self.rid, 0), findings_chars=len(findings))
        return findings


def _build_supervisor(ctx: RunContext, brief: str, findings_store: list[dict]) -> Agent:
    s, b = ctx.settings, ctx.req.budget
    sem = threading.Semaphore(b["max_parallel"])
    counter = {"n": 0}
    lock = threading.Lock()

    @tool
    def conduct_research(subtopic: str) -> str:
        """Delegate a self-contained research subtopic to a new researcher and get back cited findings.

        Args:
            subtopic: Fully self-contained description of what to research (entities, time window,
                exactly which facts/figures to find).
        """
        state, reason = ctx.limit_state()
        if state:
            return (f"RUN_LIMIT_REACHED ({reason}). No more researchers can be hired. "
                    "Stop delegating and finish with your coverage note.")
        with lock:
            if counter["n"] >= b["max_researchers"]:
                return (f"LIMIT: all {b['max_researchers']} researchers have been used. "
                        "Stop delegating and finish with your coverage note.")
            counter["n"] += 1
            rid = f"R{counter['n']}"
        with sem:
            try:
                findings = Researcher(ctx, rid, subtopic).run()
            except Exception as e:
                ctx.log_event(kind="researcher_error", agent=rid, error=f"{type(e).__name__}: {e}",
                              tb=traceback.format_exc()[-2000:])
                return f"Researcher {rid} failed: {e}"
        findings_store.append({"researcher": rid, "subtopic": subtopic, "findings": findings})
        return f"Findings from researcher {rid}:\n{findings}"

    return Agent(
        model=_model(s, s.model_id),
        tools=[conduct_research],
        system_prompt=prompts.SUPERVISOR_PROMPT.format(
            today=dt.date.today().isoformat(), brief=brief, max_parallel=b["max_parallel"],
            max_researchers=b["max_researchers"], max_rounds=b["max_supervisor_rounds"]),
        callback_handler=None,
        name="supervisor",
        hooks=[CostMeter(ctx, "supervisor", enforce=True)],
    )


_CITE_RE = re.compile(r"\[(S\d+)\]")


def _finalize_report(ctx: RunContext, report: str) -> tuple[str, dict]:
    cited = []
    for sid in _CITE_RE.findall(report):
        if sid not in cited:
            cited.append(sid)
    known = {s.sid: s for s in ctx.sources.values()}
    valid = [sid for sid in cited if sid in known]
    invalid = [sid for sid in cited if sid not in known]
    lines = ["", "## Sources", ""]
    for sid in sorted(valid, key=lambda x: int(x[1:])):
        src = known[sid]
        tag = "read" if src.fetched else "snippet-only"
        date = f", {src.published}" if src.published else ""
        lines.append(f"- [{sid}] {src.title or src.url} — {src.url} ({tag}{date})")
    if invalid:
        lines.append(f"\n> Unresolvable citation IDs in report: {', '.join(invalid)}")
    stats = {
        "citations_total": len(_CITE_RE.findall(report)),
        "unique_sources_cited": len(valid),
        "invalid_citation_ids": invalid,
        "cited_sources_read_fully": sum(1 for sid in valid if known[sid].fetched),
        "cited_sources_snippet_only": sum(1 for sid in valid if not known[sid].fetched),
        "cited_unique_domains": len({domain_of(known[sid].url) for sid in valid}),
        "cited_pdfs": sum(1 for sid in valid if known[sid].content_type == "pdf"),
        "report_chars": len(report),
    }
    return report.rstrip() + "\n" + "\n".join(lines) + "\n", stats


def run_research(req: RunRequest, settings: Optional[Settings] = None, ctx: Optional[RunContext] = None) -> dict:
    settings = settings or Settings()
    req.validate()
    ctx = ctx or RunContext(req, settings)
    store = ResultStore(settings, ctx.run_id)
    today = dt.date.today().isoformat()
    findings: list[dict] = []
    ctx.status = "running"
    store.put("request.json", {**req.to_dict(), "settings": {k: v for k, v in vars(settings).items()}})
    try:
        # 1) brief
        fq = "\n".join(f"- {q}" for q in req.focus_questions) or "- (none given)"
        brief_agent = Agent(model=_model(settings, settings.model_id, 4000), callback_handler=None, name="brief",
                            hooks=[CostMeter(ctx, "brief")])
        bres = brief_agent(prompts.BRIEF_PROMPT.format(
            today=today, topic=req.topic, audience=req.audience, language=req.language,
            focus_questions=fq, report_instructions=req.report_instructions or "(none)"))
        brief = _text_of(bres)
        if req.focus_questions:
            brief += "\n\nMust-answer questions:\n" + fq
        store.put("brief.md", brief)

        # 2) supervisor + researchers
        sup = _build_supervisor(ctx, brief, findings)
        sres = sup("Coordinate the research now.")
        store.put("supervisor_note.md", _text_of(sres))
        store.put("findings.json", findings)

        # 3) report
        src_list = "\n".join(f"[{s.sid}] {s.title} | {s.url} | {s.published or 'n/a'}"
                             for s in sorted(ctx.sources.values(), key=lambda x: int(x.sid[1:])))
        all_findings = "\n\n".join(f"### {f['researcher']}: {f['subtopic']}\n{f['findings']}" for f in findings)
        writer = Agent(model=_model(settings, settings.report_model_id, 32000), callback_handler=None, name="writer",
                       hooks=[CostMeter(ctx, "report")])
        extra = req.report_instructions or ""
        if ctx.limits_hit:
            extra += (" Research was cut short by a run limit (" + ", ".join(ctx.limits_hit) +
                      "); say so in Limitations.")
        rres = writer(prompts.REPORT_PROMPT.format(
            today=today, brief=brief, findings=all_findings or "(no findings)", sources=src_list or "(none)",
            language=req.language, audience=req.audience, report_instructions=extra))
        report, cite_stats = _finalize_report(ctx, _text_of(rres))
        ctx.artifacts["report"] = store.put("report.md", report)
        ctx.status = "completed"
    except Exception as e:
        ctx.status = "failed"
        ctx.error = f"{type(e).__name__}: {e}"
        cite_stats = {}
        log.exception("run failed")
        store.put("error.txt", traceback.format_exc())
    finally:
        ctx.artifacts["sources"] = store.put("sources.json", ctx.sources_dump())
        ctx.artifacts["events"] = store.put("events.json", ctx.events)
    summary = {**ctx.summary(), "citation_stats": cite_stats}
    store.put("summary.json", summary)  # written last: the eval harness polls for this file
    return summary
