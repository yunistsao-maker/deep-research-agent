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
        "name": "search_plus_browser",
        "tools": ["web_search", "browser_search", "browser_open", "browser_click", "browser_read"],
        "desc": "AgentCore Web Search + AgentCore Browser (managed Chromium). The agent can search through a search engine in the browser, open JS-rendered pages and click through.",
    },
}

# ---------------------------------------------------------------------------
# Depth budgets (identical across modes)
# ---------------------------------------------------------------------------
DEPTHS = {
    #            researchers  tool calls/researcher  supervisor rounds  parallel
    "quick":    dict(max_researchers=3, max_tool_calls=8,  max_supervisor_rounds=2, max_parallel=3),
    "standard": dict(max_researchers=5, max_tool_calls=12, max_supervisor_rounds=3, max_parallel=3),
    "deep":     dict(max_researchers=8, max_tool_calls=20, max_supervisor_rounds=4, max_parallel=4),
}


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

    def validate(self) -> None:
        if not self.topic or not self.topic.strip():
            raise ValueError("topic is required")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        if self.depth not in DEPTHS:
            raise ValueError(f"depth must be one of {list(DEPTHS)}")

    @property
    def budget(self) -> dict:
        return DEPTHS[self.depth]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict) -> "RunRequest":
        known = {k: payload[k] for k in cls.__dataclass_fields__ if k in payload}
        req = cls(**known)
        req.mode = req.mode.upper()
        req.validate()
        return req
