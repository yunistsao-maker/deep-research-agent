"""Per-run state: source registry, tool-call telemetry, budgets and result storage.

Everything the evaluation needs is recorded here:
  * every tool call (who, what, args, latency, outcome, bytes, URLs surfaced)
  * every source the agent has *seen* (with the exact evidence text it saw)
  * token usage per pipeline stage
so that the eval can separate "what the tool returned" from "what the model wrote".
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import boto3

from .config import RunRequest, Settings


def normalize_url(url: str) -> str:
    """Canonical form used for the S2 allowlist and de-duplication."""
    try:
        p = urlsplit(url.strip())
        scheme = "https" if p.scheme in ("http", "https") else p.scheme
        netloc = p.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = p.path.rstrip("/") or "/"
        return urlunsplit((scheme, netloc, path, p.query, ""))
    except Exception:
        return url.strip()


def domain_of(url: str) -> str:
    try:
        d = urlsplit(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""


@dataclass
class Source:
    sid: str
    url: str
    title: str = ""
    published: str = ""
    first_seen_via: str = ""      # web_search | fetch_url | browser_*
    evidence: list[str] = field(default_factory=list)  # text the agent actually saw (snippets / page text)
    fetched: bool = False         # full page content was read (fetch or browser)
    content_type: str = ""        # html | pdf | ...


class BudgetExceeded(Exception):
    pass


class RunContext:
    def __init__(self, req: RunRequest, settings: Settings):
        self.req = req
        self.settings = settings
        self.run_id = req.run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        req.run_id = self.run_id
        self.started_at = time.time()
        self._lock = threading.Lock()
        self.sources: dict[str, Source] = {}        # normalized url -> Source
        self._sid_counter = 0
        self.search_result_urls: set[str] = set()   # S2 allowlist (normalized)
        self.events: list[dict[str, Any]] = []
        self.tool_calls_by_agent: dict[str, int] = {}
        self.usage: dict[str, dict[str, int]] = {}   # stage -> token usage (kept live, see sync_usage)
        self._usage_seen: dict[int, dict[str, int]] = {}  # agent id -> accumulated usage already counted
        self.cost_cap_usd, self.minutes_cap = req.caps(settings)
        self.limits_hit: dict[str, str] = {}         # {"cost": "soft"|"hard", "time": "soft"|"hard"}
        self.status = "created"
        self.error: Optional[str] = None
        self.artifacts: dict[str, str] = {}

    # ------------------------------------------------------------------ sources
    def register_source(self, url: str, title: str = "", published: str = "", via: str = "",
                        evidence: str = "", fetched: bool = False, content_type: str = "") -> Source:
        key = normalize_url(url)
        with self._lock:
            src = self.sources.get(key)
            if src is None:
                self._sid_counter += 1
                src = Source(sid=f"S{self._sid_counter}", url=url, title=title or "", published=published or "",
                             first_seen_via=via)
                self.sources[key] = src
            if title and not src.title:
                src.title = title
            if published and not src.published:
                src.published = published
            if evidence:
                # keep evidence bounded; this is what the judge uses for "grounded in what it saw"
                if sum(len(e) for e in src.evidence) < 40000:
                    src.evidence.append(evidence[:20000])
            if fetched:
                src.fetched = True
            if content_type:
                src.content_type = content_type
            return src

    def add_search_results(self, urls: list[str]) -> None:
        with self._lock:
            for u in urls:
                self.search_result_urls.add(normalize_url(u))

    def url_allowed_in_scope(self, url: str) -> bool:
        return normalize_url(url) in self.search_result_urls

    def source_by_sid(self, sid: str) -> Optional[Source]:
        for s in self.sources.values():
            if s.sid == sid:
                return s
        return None

    # ------------------------------------------------------------------ telemetry
    def charge_tool_call(self, agent_id: str) -> int:
        """Increment and enforce the per-researcher tool budget (identical across modes) and the run caps."""
        state, reason = self.limit_state()
        if state:
            raise BudgetExceeded(
                f"RUN_LIMIT_REACHED ({reason}). Do not call more tools; write up your findings now "
                "with the sources you already have.")
        limit = self.req.budget["max_tool_calls"]
        with self._lock:
            n = self.tool_calls_by_agent.get(agent_id, 0) + 1
            self.tool_calls_by_agent[agent_id] = n
        if n > limit:
            raise BudgetExceeded(
                f"BUDGET_EXHAUSTED: you have used all {limit} tool calls. Do not call more tools; "
                "write up your findings now with the sources you already have.")
        return n

    def log_event(self, **ev: Any) -> None:
        ev.setdefault("ts", round(time.time() - self.started_at, 3))
        with self._lock:
            self.events.append(ev)

    _USAGE_KEYS = ("inputTokens", "outputTokens", "totalTokens", "cacheReadInputTokens", "cacheWriteInputTokens")

    def sync_usage(self, stage: str, agent: Any, final: bool = False) -> None:
        """Fold an agent's accumulated usage into the run total (only the part not counted yet).

        Called before every model call (live cost for the caps) and once when the agent finishes.
        """
        try:
            acc = agent.event_loop_metrics.accumulated_usage or {}
        except Exception:
            acc = {}
        key = id(agent)
        with self._lock:
            seen = self._usage_seen.get(key, {})
            agg = self.usage.setdefault(stage, {k: 0 for k in self._USAGE_KEYS} | {"calls": 0})
            for k in self._USAGE_KEYS:
                agg[k] += max(int(acc.get(k, 0) or 0) - int(seen.get(k, 0) or 0), 0)
            self._usage_seen[key] = {k: int(acc.get(k, 0) or 0) for k in self._USAGE_KEYS}
            if final:
                agg["calls"] += 1

    def _billable_searches(self) -> int:
        return sum(1 for e in self.events
                   if e.get("kind") == "tool" and e.get("tool") == "web_search" and e.get("outcome") in ("ok", "empty"))

    def spent_usd(self) -> tuple[float, float]:
        """(model USD, web search USD) so far, from list prices in Settings."""
        with self._lock:
            model = sum(self.settings.model_cost(u) for u in self.usage.values())
        return model, self._billable_searches() * self.settings.price_web_search_per_1k / 1000

    def limit_state(self) -> tuple[Optional[str], str]:
        """None while under SOFT_LIMIT_SHARE of both caps; "soft" past it (stop researching);
        "hard" at the cap itself (stop researcher/supervisor model calls; report is still written)."""
        from .config import SOFT_LIMIT_SHARE
        model, search = self.spent_usd()
        spent = model + search
        minutes = (time.time() - self.started_at) / 60
        state, reason = None, ""
        for kind, used, cap, unit in (("cost", spent, self.cost_cap_usd, "USD"),
                                      ("time", minutes, self.minutes_cap, "min")):
            level = "hard" if used >= cap else ("soft" if used >= cap * SOFT_LIMIT_SHARE else None)
            if not level:
                continue
            if self.limits_hit.get(kind) != level and not (self.limits_hit.get(kind) == "hard"):
                self.limits_hit[kind] = level
                self.log_event(kind="limit", limit=kind, level=level, used=round(used, 3), cap=cap)
            if state != "hard":
                state = level
                reason = f"{kind} limit: {used:.2f} of {cap:g} {unit} used"
        return state, reason

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict[str, Any]:
        tool_counts: dict[str, int] = {}
        outcome_counts: dict[str, int] = {}
        fetch_bytes = 0
        for e in self.events:
            if e.get("kind") != "tool":
                continue
            tool_counts[e["tool"]] = tool_counts.get(e["tool"], 0) + 1
            key = f'{e["tool"]}:{e.get("outcome", "ok")}'
            outcome_counts[key] = outcome_counts.get(key, 0) + 1
            fetch_bytes += int(e.get("bytes", 0) or 0)
        srcs = list(self.sources.values())
        total_usage = {k: sum(v.get(k, 0) for v in self.usage.values())
                       for k in ("inputTokens", "outputTokens", "totalTokens", "cacheReadInputTokens",
                                 "cacheWriteInputTokens")}
        s = self.settings
        model_usd, search_usd = self.spent_usd()
        cached_in = total_usage["cacheReadInputTokens"]
        all_in = total_usage["inputTokens"] + cached_in + total_usage["cacheWriteInputTokens"]
        return {
            "run_id": self.run_id,
            "run_label": self.req.run_label,
            "mode": self.req.mode,
            "depth": self.req.depth,
            "topic": self.req.topic,
            "status": self.status,
            "error": self.error,
            "elapsed_s": round(time.time() - self.started_at, 1),
            "model_id": self.settings.model_id,
            "tool_calls": tool_counts,
            "tool_outcomes": outcome_counts,
            "web_search_queries": tool_counts.get("web_search", 0),
            "browser_actions": sum(v for k, v in tool_counts.items() if k.startswith("browser_")),
            "fetch_bytes": fetch_bytes,
            "sources_seen": len(srcs),
            "sources_read_fully": sum(1 for s in srcs if s.fetched),
            "unique_domains_seen": len({domain_of(s.url) for s in srcs}),
            "pdf_sources_read": sum(1 for s in srcs if s.fetched and s.content_type == "pdf"),
            "researcher_tool_calls": self.tool_calls_by_agent,
            "token_usage_by_stage": self.usage,
            "token_usage_total": total_usage,
            "prompt_cache": s.prompt_cache,
            "cache_hit_ratio": round(cached_in / all_in, 3) if all_in else None,
            # estimate from list prices in Settings; excludes Runtime/Browser compute (cents per run)
            "est_cost_usd": {"model": round(model_usd, 3), "web_search": round(search_usd, 3),
                             "total": round(model_usd + search_usd, 3)},
            "caps": {"max_cost_usd": self.cost_cap_usd, "max_minutes": self.minutes_cap},
            # non-empty = the run was cut short by a cap; the eval flags these runs
            "limits_hit": self.limits_hit,
            "artifacts": self.artifacts,
        }

    def sources_dump(self) -> list[dict[str, Any]]:
        return [{"sid": s.sid, "url": s.url, "title": s.title, "published": s.published,
                 "first_seen_via": s.first_seen_via, "fetched": s.fetched, "content_type": s.content_type,
                 "domain": domain_of(s.url), "evidence": s.evidence}
                for s in sorted(self.sources.values(), key=lambda x: int(x.sid[1:]))]


class ResultStore:
    """Writes run artifacts to S3 (if RESULTS_BUCKET is set) and always to local disk."""

    def __init__(self, settings: Settings, run_id: str):
        self.settings = settings
        self.run_id = run_id
        self.local_dir = os.path.join(settings.local_results_dir, run_id)
        os.makedirs(self.local_dir, exist_ok=True)
        self.s3 = boto3.client("s3", region_name=settings.region) if settings.results_bucket else None

    def key(self, name: str) -> str:
        return f"{self.settings.results_prefix}/{self.run_id}/{name}"

    def put(self, name: str, data: str | dict | list, content_type: Optional[str] = None) -> str:
        if not isinstance(data, str):
            data = json.dumps(data, ensure_ascii=False, indent=2, default=str)
            content_type = content_type or "application/json"
        content_type = content_type or ("text/markdown" if name.endswith(".md") else "text/plain")
        path = os.path.join(self.local_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
        if self.s3:
            self.s3.put_object(Bucket=self.settings.results_bucket, Key=self.key(name),
                               Body=data.encode("utf-8"), ContentType=f"{content_type}; charset=utf-8")
            return f"s3://{self.settings.results_bucket}/{self.key(name)}"
        return path
