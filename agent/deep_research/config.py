"""Run configuration: tool-setting modes, depth budgets and environment settings.

The four modes differ ONLY in which retrieval tools the researcher sub-agents get.
Model, prompts (except the tool-usage paragraph), budgets and pipeline are identical,
so differences in output can be attributed to the retrieval setting.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Modes (the experimental variable)
# ---------------------------------------------------------------------------
MODES = {
    "S1": {
        "name": "search_index_only",
        "tools": ["web_search"],
        "desc": "AgentCore Web Search only. The agent sees titles, URLs, dates and semantic snippets; it never opens a page.",
    },
    "S2": {
        "name": "search_plus_fetch_scoped",
        "tools": ["web_search", "fetch_url"],
        "fetch_scope": "search_results",
        "desc": "AgentCore Web Search + fetch_url, but fetch_url only accepts URLs that a web_search call in this run returned.",
    },
    "S3": {
        "name": "search_plus_fetch_open",
        "tools": ["web_search", "fetch_url"],
        "fetch_scope": "any",
        "desc": "AgentCore Web Search + fetch_url on any public URL (can follow links found inside fetched pages, e.g. PDFs).",
    },
    "S4": {
        "name": "search_plus_browser_search",
        "tools": ["web_search", "browser_search"],
        "desc": "AgentCore Web Search + search through AgentCore Browser on a public search engine (Bing). "
                "The agent sees result titles, URLs and snippets from both; it never opens a page and has no fetch.",
    },
    # Optional, not in the default matrix: the full browser (opens and reads pages, clicks through).
    "S5": {
        "name": "search_plus_browser_full",
        "tools": ["web_search", "browser_search", "browser_open", "browser_click", "browser_read"],
        "desc": "AgentCore Web Search + AgentCore Browser that can search, open JS-rendered pages, click and read them.",
    },
}

# ---------------------------------------------------------------------------
# Budget modes (second experimental factor): how a researcher is told to stop
# ---------------------------------------------------------------------------
BUDGET_MODES = {
    "none": "Tool-call budget only (max_tool_calls). No token budget; every mode gets the same number of calls.",
    "native": "Claude task budget (output_config.task_budget, beta task-budgets-2026-03-13) via Bedrock InvokeModel. "
              "The model sees a server-side countdown and paces itself.",
    "client": "Same token budget, counted by this agent. The remaining budget is appended to every tool result, "
              "with a wrap-up instruction at 70% and a stop instruction at 90%; tools refuse at 100%.",
}
TASK_BUDGET_BETA = "task-budgets-2026-03-13"
TASK_BUDGET_MIN = 20000            # API minimum
# In native/client modes the token budget is the binding limit; the tool-call limit is raised to this
# multiple of max_tool_calls and only stops runaways.
BUDGET_MODE_TOOL_CALL_FACTOR = 2
# Backstop: researcher model calls are cancelled when counted tokens exceed budget x this factor
# (native mode: the model ignored the countdown; client mode: it ignored the stop instruction).
BUDGET_OVERRUN_FACTOR = {"native": 1.5, "client": 1.15}

# ---------------------------------------------------------------------------
# Depth budgets (identical across modes)
# ---------------------------------------------------------------------------
DEPTHS = {
    #            researchers  tool calls/researcher  supervisor rounds  parallel
    #            max_cost_usd / max_minutes = per-run caps (about 2-3x the expected cost/time with prompt caching).
    #            They are a safety net, not the experimental budget: if a mode hits them often, raise them,
    #            otherwise the cap (not the retrieval setting) decides the result.
    "quick":    dict(max_researchers=3, max_tool_calls=8,  max_supervisor_rounds=2, max_parallel=3,
                     max_cost_usd=3.0, max_minutes=20, researcher_token_budget=40000),
    "standard": dict(max_researchers=5, max_tool_calls=12, max_supervisor_rounds=3, max_parallel=3,
                     max_cost_usd=8.0, max_minutes=45, researcher_token_budget=60000),
    "deep":     dict(max_researchers=8, max_tool_calls=20, max_supervisor_rounds=4, max_parallel=4,
                     max_cost_usd=20.0, max_minutes=90, researcher_token_budget=100000),
}
# Research stops (no new tool calls / researchers) at this share of the cap, keeping the rest for
# compression + the final report, so a capped run still ends with a report.
SOFT_LIMIT_SHARE = 0.8


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


@dataclass
class Settings:
    """Deployment-level settings read from environment variables."""

    region: str = field(default_factory=lambda: _env("AWS_REGION", "us-west-2"))
    # Same model everywhere by default so the retrieval setting is the only variable.
    model_id: str = field(default_factory=lambda: _env("MODEL_ID", "global.anthropic.claude-sonnet-5-5"))
    compress_model_id: str = field(default_factory=lambda: _env("COMPRESS_MODEL_ID", _env("MODEL_ID", "global.anthropic.claude-sonnet-5-5")))
    report_model_id: str = field(default_factory=lambda: _env("REPORT_MODEL_ID", _env("MODEL_ID", "global.anthropic.claude-sonnet-5-5")))
    max_tokens: int = field(default_factory=lambda: int(_env("MAX_TOKENS", "16000")))
    # Bedrock prompt caching (default on). PROMPT_CACHE=false to measure uncached cost.
    prompt_cache: bool = field(default_factory=lambda: _env("PROMPT_CACHE", "true").lower() == "true")
    # USD per 1M tokens, used only for the cost estimate in summary.json.
    # Defaults = Claude Sonnet 5.5, Bedrock global profile (regional `us.` profiles are +10%).
    price_input: float = field(default_factory=lambda: float(_env("PRICE_INPUT_PER_MTOK", "2.0")))
    price_output: float = field(default_factory=lambda: float(_env("PRICE_OUTPUT_PER_MTOK", "10.0")))
    price_cache_write: float = field(default_factory=lambda: float(_env("PRICE_CACHE_WRITE_PER_MTOK", "2.5")))
    price_cache_read: float = field(default_factory=lambda: float(_env("PRICE_CACHE_READ_PER_MTOK", "0.10")))
    price_web_search_per_1k: float = field(default_factory=lambda: float(_env("PRICE_WEB_SEARCH_PER_1K", "7.0")))
    # budget_mode "native": anthropic_version sent to Bedrock InvokeModel. Empty = the Anthropic SDK's default.
    # If Bedrock rejects it (HTTP 400 naming anthropic_version), set NATIVE_ANTHROPIC_VERSION=bedrock-2023-05-31.
    native_anthropic_version: Optional[str] = field(default_factory=lambda: _env("NATIVE_ANTHROPIC_VERSION"))
    # Hard ceilings set on the runtime; a payload's max_cost_usd / max_minutes can never exceed them.
    max_cost_ceiling_usd: float = field(default_factory=lambda: float(_env("MAX_COST_PER_RUN_USD", "30")))
    max_minutes_ceiling: float = field(default_factory=lambda: float(_env("MAX_MINUTES_PER_RUN", "120")))

    def model_cost(self, usage: dict) -> float:
        """USD for a Bedrock usage dict (inputTokens excludes cached tokens on Bedrock)."""
        return (int(usage.get("inputTokens", 0) or 0) * self.price_input
                + int(usage.get("outputTokens", 0) or 0) * self.price_output
                + int(usage.get("cacheWriteInputTokens", 0) or 0) * self.price_cache_write
                + int(usage.get("cacheReadInputTokens", 0) or 0) * self.price_cache_read) / 1e6
    # AgentCore Gateway (MCP endpoint with the web-search connector target, IAM inbound auth)
    gateway_url: Optional[str] = field(default_factory=lambda: _env("GATEWAY_URL"))
    web_search_tool_name: Optional[str] = field(default_factory=lambda: _env("WEB_SEARCH_TOOL_NAME"))  # auto-discovered if empty
    # Fetch tool
    fetch_timeout_s: float = field(default_factory=lambda: float(_env("FETCH_TIMEOUT_S", "20")))
    fetch_max_chars: int = field(default_factory=lambda: int(_env("FETCH_MAX_CHARS", "15000")))
    fetch_max_bytes: int = field(default_factory=lambda: int(_env("FETCH_MAX_BYTES", str(15 * 1024 * 1024))))
    fetch_user_agent: str = field(default_factory=lambda: _env(
        "FETCH_USER_AGENT", "Mozilla/5.0 (compatible; DeepResearchEval/1.0; +https://aws.amazon.com/bedrock/agentcore/)"))
    respect_robots: bool = field(default_factory=lambda: _env("RESPECT_ROBOTS", "true").lower() == "true")
    # Browser tool
    browser_identifier: str = field(default_factory=lambda: _env("BROWSER_IDENTIFIER", "aws.browser.v1"))
    browser_search_engine: str = field(default_factory=lambda: _env("BROWSER_SEARCH_ENGINE", "bing"))  # bing | duckduckgo | google
    browser_page_chars: int = field(default_factory=lambda: int(_env("BROWSER_PAGE_CHARS", "15000")))
    # Results
    results_bucket: Optional[str] = field(default_factory=lambda: _env("RESULTS_BUCKET"))
    results_prefix: str = field(default_factory=lambda: _env("RESULTS_PREFIX", "deep-research-runs"))
    local_results_dir: str = field(default_factory=lambda: _env("LOCAL_RESULTS_DIR", "/tmp/deep-research-runs"))


@dataclass
class RunRequest:
    """Dynamic input for one research run (what you send in the invoke payload)."""

    topic: str
    mode: str = "S1"
    depth: str = "standard"
    language: str = "zh-TW"                 # report language
    audience: str = "enterprise solutions architects"
    focus_questions: list[str] = field(default_factory=list)   # must-answer questions (probes)
    published_after: Optional[str] = None   # ISO-8601, passed to web_search publishedDateFilter.from
    include_domains: list[str] = field(default_factory=list)
    exclude_domains: list[str] = field(default_factory=list)
    report_instructions: str = ""           # extra format requirements
    run_id: Optional[str] = None
    run_label: Optional[str] = None         # e.g. "dc-infra/S3/rep1" for the eval matrix
    max_cost_usd: Optional[float] = None    # per-run cost cap; default = DEPTHS[depth]["max_cost_usd"]
    max_minutes: Optional[float] = None     # per-run wall-clock cap; default = DEPTHS[depth]["max_minutes"]
    budget_mode: str = "none"               # none | native | client  (see BUDGET_MODES)
    task_budget_tokens: Optional[int] = None  # per-researcher token budget; default by depth

    def validate(self) -> None:
        if not self.topic or not self.topic.strip():
            raise ValueError("topic is required")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        if self.depth not in DEPTHS:
            raise ValueError(f"depth must be one of {list(DEPTHS)}")
        if self.budget_mode not in BUDGET_MODES:
            raise ValueError(f"budget_mode must be one of {list(BUDGET_MODES)}")
        if self.task_budget_tokens is not None and int(self.task_budget_tokens) < TASK_BUDGET_MIN:
            raise ValueError(f"task_budget_tokens must be >= {TASK_BUDGET_MIN}")
        for name in ("max_cost_usd", "max_minutes"):
            v = getattr(self, name)
            if v is not None and float(v) <= 0:
                raise ValueError(f"{name} must be > 0")

    @property
    def budget(self) -> dict:
        return DEPTHS[self.depth]

    @property
    def token_budget(self) -> Optional[int]:
        """Per-researcher token budget, or None in budget_mode 'none'. Same value for every retrieval mode."""
        if self.budget_mode == "none":
            return None
        return int(self.task_budget_tokens or self.budget["researcher_token_budget"])

    @property
    def tool_call_limit(self) -> int:
        n = self.budget["max_tool_calls"]
        return n if self.budget_mode == "none" else n * BUDGET_MODE_TOOL_CALL_FACTOR

    def caps(self, settings: "Settings") -> tuple[float, float]:
        """Effective (cost USD, minutes) caps: request value or depth default, never above the runtime ceiling."""
        cost = float(self.max_cost_usd if self.max_cost_usd is not None else self.budget["max_cost_usd"])
        mins = float(self.max_minutes if self.max_minutes is not None else self.budget["max_minutes"])
        return min(cost, settings.max_cost_ceiling_usd), min(mins, settings.max_minutes_ceiling)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict) -> "RunRequest":
        known = {k: payload[k] for k in cls.__dataclass_fields__ if k in payload}
        req = cls(**known)
        req.mode = req.mode.upper()
        req.budget_mode = (req.budget_mode or "none").lower()
        req.validate()
        return req
