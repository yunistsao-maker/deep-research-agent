"""AgentCore Web Search (managed connector on AgentCore Gateway), wrapped as a Strands tool.

We call the gateway's MCP tool ourselves instead of handing the raw MCP tool to the model, so that
  * every query and every returned URL is recorded (needed for the S2 allowlist and the eval),
  * results are rendered with stable source IDs ([S12]) the agent must cite,
  * request-level filters (domains, published date) from the run request are applied uniformly.

Gateway inbound auth is AWS_IAM: the runtime execution role signs requests with SigV4.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Optional

from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
from strands import tool
from strands.tools.mcp import MCPClient

from ..config import Settings
from ..run_context import BudgetExceeded, RunContext

_client_lock = threading.Lock()
_client: Optional[MCPClient] = None
_tool_name: Optional[str] = None


def _get_client(settings: Settings) -> tuple[MCPClient, str]:
    """One long-lived MCP session to the gateway per process (thread-safe)."""
    global _client, _tool_name
    with _client_lock:
        if _client is None:
            if not settings.gateway_url:
                raise RuntimeError("GATEWAY_URL is not set (AgentCore Gateway MCP URL with the web-search target)")
            client = MCPClient(lambda: aws_iam_streamablehttp_client(
                endpoint=settings.gateway_url, aws_region=settings.region, aws_service="bedrock-agentcore"))
            client.start()
            name = settings.web_search_tool_name
            if not name:
                tools = client.list_tools_sync()
                names = [t.tool_name for t in tools]
                cands = [n for n in names if n.endswith("___WebSearch") or n == "WebSearch"]
                if not cands:
                    raise RuntimeError(f"No WebSearch tool found on gateway; tools = {names}")
                name = cands[0]
            _client, _tool_name = client, name
        return _client, _tool_name


def raw_web_search(settings: Settings, query: str, max_results: int = 10,
                   include_domains: Optional[list[str]] = None, exclude_domains: Optional[list[str]] = None,
                   published_after: Optional[str] = None, published_before: Optional[str] = None) -> list[dict]:
    client, name = _get_client(settings)
    args: dict[str, Any] = {"query": query[:200], "maxResults": max(1, min(int(max_results), 25))}
    filters: dict[str, Any] = {}
    dom: dict[str, Any] = {}
    if include_domains:
        dom["include"] = include_domains[:100]
    if exclude_domains:
        dom["exclude"] = exclude_domains[:100]
    if dom:
        filters["domainFilter"] = dom
    if published_after or published_before:
        pdf: dict[str, str] = {}
        if published_after:
            pdf["from"] = published_after
        if published_before:
            pdf["to"] = published_before
        filters["publishedDateFilter"] = pdf
    if filters:
        args["filters"] = filters
    res = client.call_tool_sync(tool_use_id=f"ws-{uuid.uuid4().hex[:8]}", name=name, arguments=args)
    if res.get("status") == "error" and "filters" in args:
        # request-level filters need connector version >= 1.2.0; retry without them on older targets
        args.pop("filters")
        res = client.call_tool_sync(tool_use_id=f"ws-{uuid.uuid4().hex[:8]}", name=name, arguments=args)
    if res.get("status") == "error":
        raise RuntimeError(f"web search error: {res.get('content')}")
    results: list[dict] = []
    for block in res.get("content", []) or []:
        text = block.get("text") if isinstance(block, dict) else None
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        for r in payload.get("results", []) or []:
            results.append({
                "url": r.get("url"),
                "title": r.get("title"),
                "published": r.get("publishedDate"),
                "text": r.get("text") or "",
            })
    return results


def make_web_search_tool(ctx: RunContext, agent_id: str):
    settings = ctx.settings
    req = ctx.req

    @tool
    def web_search(query: str, max_results: int = 10, include_domains: Optional[list[str]] = None,
                   published_after: Optional[str] = None) -> str:
        """Search the web (Amazon-operated web index via AgentCore Web Search).

        Returns up to max_results results, each with a source ID like [S12], title, URL, publication date
        and a short query-relevant snippet. Cite facts with the source ID.

        Args:
            query: Natural-language search query, max 200 characters. Prefer specific queries
                (entity + attribute + year) over broad ones.
            max_results: 1-25, default 10.
            include_domains: Optional list of domains to restrict this search to (e.g. ["ocp.org", "tsmc.com"]).
            published_after: Optional ISO-8601 date (YYYY-MM-DD); only results published on/after it.
        """
        t0 = time.time()
        try:
            ctx.charge_tool_call(agent_id)
        except BudgetExceeded as e:
            ctx.log_event(kind="tool", tool="web_search", agent=agent_id, outcome="budget", query=query)
            return str(e)
        incl = include_domains or req.include_domains or None
        after = published_after or req.published_after
        if after and len(after) == 10:
            after = after + "T00:00:00Z"
        try:
            results = raw_web_search(settings, query, max_results, incl, req.exclude_domains or None, after)
        except Exception as e:  # surface the error to the model, keep going
            ctx.log_event(kind="tool", tool="web_search", agent=agent_id, outcome="error", query=query,
                          error=str(e)[:500], latency=round(time.time() - t0, 2))
            return f"web_search failed: {e}"
        urls = [r["url"] for r in results if r.get("url")]
        ctx.add_search_results(urls)
        lines = [f'web_search results for "{query}" ({len(results)} results):']
        for r in results:
            if r.get("url"):
                src = ctx.register_source(r["url"], r.get("title") or "", r.get("published") or "",
                                          via="web_search", evidence=r.get("text", ""))
                lines.append(f'\n[{src.sid}] {r.get("title") or "(no title)"}\nURL: {r["url"]}\n'
                             f'Published: {r.get("published") or "unknown"}\nSnippet: {r.get("text", "").strip()}')
            else:  # knowledge-graph observation (no URL)
                lines.append(f'\n[KG] (knowledge-graph fact, no URL; corroborate before citing)\n{r.get("text", "").strip()}')
        ctx.log_event(kind="tool", tool="web_search", agent=agent_id, outcome="ok" if results else "empty",
                      query=query, n_results=len(results), urls=urls, latency=round(time.time() - t0, 2),
                      bytes=sum(len(r.get("text", "")) for r in results))
        return "\n".join(lines)

    return web_search
