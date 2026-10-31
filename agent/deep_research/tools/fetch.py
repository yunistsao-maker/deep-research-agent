"""fetch_url tool: plain HTTP GET + main-content extraction (HTML via trafilatura, PDF via pypdf).

S2 (scope="search_results"): only URLs returned by web_search earlier in this run are allowed.
S3 (scope="any"):            any public http(s) URL, so the agent can follow links found inside pages.

The tool output is identical in S2 and S3 (including the list of outbound links), so the only
difference between the two settings is the allowlist. Rejected attempts are logged ("blocked_scope")
which tells you how often the agent *wanted* to go beyond the search results.
"""
from __future__ import annotations

import io
import ipaddress
import socket
import threading
import time
import urllib.robotparser
from typing import Optional
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura
from lxml import html as lxml_html
from strands import tool

from ..run_context import BudgetExceeded, RunContext, domain_of, normalize_url

_robots_cache: dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
_robots_lock = threading.Lock()


def _is_public_host(host: str) -> bool:
    """Block SSRF to link-local/private ranges (e.g. 169.254.169.254, VPC internal hosts)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def _robots_allows(url: str, ua: str, timeout: float) -> bool:
    p = urlsplit(url)
    base = f"{p.scheme}://{p.netloc}"
    with _robots_lock:
        rp = _robots_cache.get(base, "missing")  # type: ignore[arg-type]
    if rp == "missing":
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = httpx.get(base + "/robots.txt", timeout=min(timeout, 8), headers={"User-Agent": ua},
                          follow_redirects=True)
            if r.status_code >= 400:
                rp = None  # no robots.txt -> allowed
            else:
                rp.parse(r.text.splitlines())
        except Exception:
            rp = None
        with _robots_lock:
            _robots_cache[base] = rp
    return True if rp is None else rp.can_fetch(ua, url)


def _extract_links(base_url: str, raw_html: str, limit: int = 30) -> list[tuple[str, str]]:
    try:
        doc = lxml_html.fromstring(raw_html)
    except Exception:
        return []
    seen, out = set(), []
    base_dom = domain_of(base_url)
    cands = []
    for a in doc.xpath("//a[@href]"):
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        absu = urljoin(base_url, href)
        if not absu.startswith(("http://", "https://")):
            continue
        n = normalize_url(absu)
        if n in seen or n == normalize_url(base_url):
            continue
        seen.add(n)
        text = " ".join((a.text_content() or "").split())[:100]
        # rank: documents first, then same-site deep links with descriptive anchor text
        score = 0
        low = absu.lower()
        if low.endswith(".pdf") or "/pdf" in low or "download" in low:
            score += 3
        if domain_of(absu) == base_dom:
            score += 1
        if len(text) > 15:
            score += 1
        cands.append((score, absu, text))
    cands.sort(key=lambda x: -x[0])
    for _, u, t in cands[:limit]:
        out.append((u, t))
    return out


def _pdf_to_text(data: bytes, max_pages: int = 40) -> tuple[str, int]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = reader.pages[:max_pages]
    parts = []
    for i, pg in enumerate(pages):
        try:
            parts.append(f"[page {i + 1}]\n" + (pg.extract_text() or ""))
        except Exception:
            continue
    return "\n".join(parts), len(reader.pages)


def make_fetch_tool(ctx: RunContext, agent_id: str, scope: str):
    s = ctx.settings
    cache: dict[str, tuple[str, list[tuple[str, str]], str, str]] = {}

    @tool
    def fetch_url(url: str, offset: int = 0) -> str:
        """Open a web page or PDF and return its main text content plus the most relevant outbound links.

        Use it to read primary sources in full (specs, whitepapers, filings, datasheets, docs) after
        web_search has located them. Long documents are paginated: call again with offset to continue.

        Args:
            url: Absolute http(s) URL.
            offset: Character offset for long documents (use the value given in the previous output).
        """
        t0 = time.time()
        if scope == "search_results" and not ctx.url_allowed_in_scope(url):
            # does NOT consume budget; logged so the eval can count "wanted to go deeper" events
            ctx.log_event(kind="tool", tool="fetch_url", agent=agent_id, outcome="blocked_scope", url=url)
            return ("REJECTED: in this configuration fetch_url can only open URLs that appeared in web_search "
                    "results during this run. Use web_search to surface this page first (e.g. search its title "
                    "or domain + topic), then fetch the URL exactly as returned.")
        try:
            ctx.charge_tool_call(agent_id)
        except BudgetExceeded as e:
            ctx.log_event(kind="tool", tool="fetch_url", agent=agent_id, outcome="budget", url=url)
            return str(e)

        key = normalize_url(url)
        if key not in cache:
            p = urlsplit(url)
            if p.scheme not in ("http", "https") or not p.hostname or not _is_public_host(p.hostname):
                ctx.log_event(kind="tool", tool="fetch_url", agent=agent_id, outcome="blocked_ssrf", url=url)
                return "REJECTED: only public http(s) URLs are allowed."
            if s.respect_robots and not _robots_allows(url, s.fetch_user_agent, s.fetch_timeout_s):
                ctx.log_event(kind="tool", tool="fetch_url", agent=agent_id, outcome="blocked_robots", url=url)
                return "REJECTED: robots.txt disallows fetching this URL. Try another source."
            try:
                with httpx.Client(timeout=s.fetch_timeout_s, follow_redirects=True, max_redirects=5,
                                  headers={"User-Agent": s.fetch_user_agent,
                                           "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
                                           "Accept-Language": "en,zh-TW;q=0.8,zh;q=0.7"}) as client:
                    with client.stream("GET", url) as r:
                        status = r.status_code
                        ctype = (r.headers.get("content-type") or "").lower()
                        buf = bytearray()
                        for chunk in r.iter_bytes():
                            buf.extend(chunk)
                            if len(buf) > s.fetch_max_bytes:
                                break
                        final_url = str(r.url)
                data = bytes(buf)
            except Exception as e:
                ctx.log_event(kind="tool", tool="fetch_url", agent=agent_id, outcome="error", url=url,
                              error=f"{type(e).__name__}: {str(e)[:300]}", latency=round(time.time() - t0, 2))
                return f"fetch_url failed ({type(e).__name__}): {e}"
            if status >= 400:
                outcome = "http_403" if status in (401, 403) else ("http_429" if status == 429 else f"http_{status}")
                ctx.log_event(kind="tool", tool="fetch_url", agent=agent_id, outcome=outcome, url=url,
                              latency=round(time.time() - t0, 2))
                return f"fetch_url: HTTP {status} for {url}. The site may block automated access; try another source."
            links: list[tuple[str, str]] = []
            kind = "html"
            if "pdf" in ctype or data[:5] == b"%PDF-":
                kind = "pdf"
                try:
                    text, npages = _pdf_to_text(data)
                    text = f"(PDF, {npages} pages; first 40 extracted)\n" + text
                except Exception as e:
                    ctx.log_event(kind="tool", tool="fetch_url", agent=agent_id, outcome="parse_error", url=url,
                                  error=str(e)[:300])
                    return f"fetch_url: could not parse PDF ({e})"
            else:
                raw = data.decode(r.encoding or "utf-8", errors="replace") if data else ""
                text = trafilatura.extract(raw, url=final_url, include_tables=True, include_links=False,
                                           favor_recall=True) or ""
                links = _extract_links(final_url, raw)
                title = ""
                try:
                    title = (lxml_html.fromstring(raw).findtext(".//title") or "").strip()
                except Exception:
                    pass
                if len(text.strip()) < 200:
                    # likely JS-rendered or blocked interstitial; report it honestly (a browser may help)
                    kind = "html_thin"
                text = (f"Title: {title}\n" if title else "") + text
            cache[key] = (text, links, kind, final_url)
        text, links, kind, final_url = cache[key]

        src = ctx.register_source(final_url, via="fetch_url", fetched=True,
                                  content_type="pdf" if kind == "pdf" else "html",
                                  evidence=text[offset: offset + s.fetch_max_chars])
        if normalize_url(final_url) != key:
            ctx.register_source(url, via="fetch_url")  # keep the original URL mapped too
        chunk = text[offset: offset + s.fetch_max_chars]
        more = offset + s.fetch_max_chars < len(text)
        out = [f"[{src.sid}] {final_url}  (type={kind}, chars {offset}-{offset + len(chunk)} of {len(text)})"]
        if kind == "html_thin":
            out.append("NOTE: very little extractable text; the page may require JavaScript or block bots.")
        out.append(chunk)
        if more:
            out.append(f"\n... [truncated; call fetch_url(url, offset={offset + s.fetch_max_chars}) for more]")
        if links and offset == 0:
            out.append("\nOutbound links (most relevant first):")
            out += [f"- {t or '(no text)'} -> {u}" for u, t in links[:20]]
        ctx.log_event(kind="tool", tool="fetch_url", agent=agent_id, outcome="thin" if kind == "html_thin" else "ok",
                      url=url, final_url=final_url, content_type=kind, chars=len(text),
                      bytes=len(chunk), latency=round(time.time() - t0, 2),
                      followed_link=not ctx.url_allowed_in_scope(url))
        return "\n".join(out)

    return fetch_url
