"""AgentCore Browser tools (S4): search through a search engine in a real browser, open JS-rendered
pages, click links and read page text.

strands_tools.browser.AgentCoreBrowser is deprecated, so we drive Playwright directly over CDP using
bedrock-agentcore's BrowserClient, as its deprecation notice recommends.

Playwright's sync API must stay on one thread, while Strands runs sync tools in worker threads,
so each researcher gets a BrowserSession that owns a dedicated thread and executes commands from a queue.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Optional
from urllib.parse import quote_plus

import trafilatura
from bedrock_agentcore.tools.browser_client import BrowserClient
from strands import tool

from ..run_context import BudgetExceeded, RunContext

SEARCH_URLS = {
    "bing": "https://www.bing.com/search?q={q}&setlang=en",
    "duckduckgo": "https://html.duckduckgo.com/html/?q={q}",
    "google": "https://www.google.com/search?q={q}&hl=en",
}
BLOCK_MARKERS = ("captcha", "unusual traffic", "are you a robot", "verify you are human",
                 "access denied", "cf-challenge", "just a moment...", "enable javascript and cookies")


class BrowserSession:
    def __init__(self, region: str, identifier: str, timeout_s: int = 1800):
        self.region, self.identifier, self.timeout_s = region, identifier, timeout_s
        self._q: "queue.Queue[tuple[Callable, queue.Queue]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._err: Optional[BaseException] = None
        self.session_id: Optional[str] = None
        self.page = None
        self.started_at: Optional[float] = None

    # ------------------------------------------------------------------ lifecycle
    def _worker(self) -> None:
        from playwright.sync_api import sync_playwright
        client = BrowserClient(region=self.region)
        try:
            self.session_id = client.start(identifier=self.identifier, session_timeout_seconds=self.timeout_s)
            ws_url, headers = client.generate_ws_headers()
            with sync_playwright() as pw:
                browser = pw.chromium.connect_over_cdp(ws_url, headers=headers)
                ctx = browser.contexts[0] if browser.contexts else browser.new_context()
                self.page = ctx.pages[0] if ctx.pages else ctx.new_page()
                self.page.set_default_timeout(30000)
                self.started_at = time.time()
                self._ready.set()
                while True:
                    fn, reply = self._q.get()
                    if fn is None:
                        break
                    try:
                        reply.put((True, fn(self.page)))
                    except Exception as e:  # return error to caller, keep session alive
                        reply.put((False, e))
                try:
                    browser.close()
                except Exception:
                    pass
        except BaseException as e:
            self._err = e
            self._ready.set()
        finally:
            try:
                client.stop()
            except Exception:
                pass

    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(target=self._worker, daemon=True, name="agentcore-browser")
        self._thread.start()
        self._ready.wait(timeout=120)
        if self._err:
            raise RuntimeError(f"could not start AgentCore Browser session: {self._err}")

    def run(self, fn: Callable[[Any], Any], timeout: float = 90) -> Any:
        self.start()
        reply: queue.Queue = queue.Queue()
        self._q.put((fn, reply))
        ok, val = reply.get(timeout=timeout)
        if not ok:
            raise val
        return val

    def close(self) -> Optional[float]:
        """Stop the session (important: idle sessions keep billing). Returns session seconds."""
        if self._thread:
            self._q.put((None, None))  # type: ignore[arg-type]
            self._thread.join(timeout=30)
        return round(time.time() - self.started_at, 1) if self.started_at else None


def _page_text(page, max_chars: int) -> tuple[str, str, str]:
    html = page.content()
    url = page.url
    title = page.title()
    text = trafilatura.extract(html, url=url, include_tables=True, favor_recall=True) or ""
    if len(text) < 200:  # fall back to rendered innerText (works for app-like pages)
        try:
            text = page.inner_text("body")[: max_chars * 2]
        except Exception:
            pass
    return url, title, text


def _links(page, limit: int = 40) -> list[tuple[str, str]]:
    js = """() => Array.from(document.querySelectorAll('a[href]')).map(a => [a.href, (a.innerText||'').trim().slice(0,100)])
              .filter(x => x[0].startsWith('http'))"""
    seen, out = set(), []
    for href, text in page.evaluate(js):
        if href in seen:
            continue
        seen.add(href)
        out.append((href, " ".join(text.split())))
        if len(out) >= limit:
            break
    return out


def make_browser_tools(ctx: RunContext, agent_id: str, session: BrowserSession):
    s = ctx.settings
    state: dict[str, Any] = {"text": "", "url": "", "sid": ""}

    def _charge(tool_name: str, **kw) -> Optional[str]:
        try:
            ctx.charge_tool_call(agent_id)
        except BudgetExceeded as e:
            ctx.log_event(kind="tool", tool=tool_name, agent=agent_id, outcome="budget", **kw)
            return str(e)
        return None

    def _render(tool_name: str, t0: float, url: str, title: str, text: str, links, via: str, **kw) -> str:
        low = (title + " " + text[:3000]).lower()
        blocked = any(m in low for m in BLOCK_MARKERS)
        src = ctx.register_source(url, title=title, via=via, fetched=not blocked,
                                  content_type="pdf" if url.lower().endswith(".pdf") else "html",
                                  evidence="" if blocked else text[: s.browser_page_chars])
        state.update(text=text, url=url, sid=src.sid)
        out = [f"[{src.sid}] {title}\nURL: {url}  (chars 0-{min(len(text), s.browser_page_chars)} of {len(text)})"]
        if blocked:
            out.append("BLOCKED: the page shows a bot check / CAPTCHA / access-denied interstitial. Do NOT try to "
                       "solve it. Go back and use another source or web_search instead.")
        out.append(text[: s.browser_page_chars])
        if len(text) > s.browser_page_chars:
            out.append(f"... [truncated; call browser_read(offset={s.browser_page_chars}) for more]")
        if links:
            out.append("\nLinks on page (use browser_click with the link text, or browser_open with the URL):")
            out += [f"- {t or '(no text)'} -> {u}" for u, t in links[:25]]
        ctx.log_event(kind="tool", tool=tool_name, agent=agent_id, outcome="blocked_bot" if blocked else "ok",
                      url=url, chars=len(text), bytes=min(len(text), s.browser_page_chars),
                      latency=round(time.time() - t0, 2), **kw)
        return "\n".join(out)

    @tool
    def browser_search(query: str, engine: Optional[str] = None) -> str:
        """Search the web through a real browser using a public search engine results page (SERP).

        Use this when web_search results look thin, stale, or miss local-language / niche sources.
        Returns the visible result page text and result links; then use browser_open or browser_click.

        Args:
            query: Search query.
            engine: "bing" (default), "duckduckgo" or "google".
        """
        t0 = time.time()
        msg = _charge("browser_search", query=query)
        if msg:
            return msg
        eng = (engine or s.browser_search_engine).lower()
        url = SEARCH_URLS.get(eng, SEARCH_URLS["bing"]).format(q=quote_plus(query))

        def go(page):
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)
            u, t, txt = _page_text(page, s.browser_page_chars)
            return u, t, txt, _links(page, 60)
        try:
            u, t, txt, links = session.run(go)
        except Exception as e:
            ctx.log_event(kind="tool", tool="browser_search", agent=agent_id, outcome="error", query=query,
                          error=str(e)[:300], latency=round(time.time() - t0, 2))
            return f"browser_search failed: {e}"
        # SERPs are navigation aids, not sources: don't register the SERP itself as citable evidence
        low = (t + " " + txt[:3000]).lower()
        blocked = any(m in low for m in BLOCK_MARKERS)
        ext = [(h, x) for h, x in links if eng not in h and "microsoft.com" not in h and "duckduckgo.com" not in h
               and "google." not in h] or links
        outcome = "blocked_bot" if blocked else ("ok" if len(ext) >= 3 else "empty")
        ctx.log_event(kind="tool", tool="browser_search", agent=agent_id, outcome=outcome,
                      query=query, engine=eng, n_links=len(ext), latency=round(time.time() - t0, 2))
        if outcome == "empty":
            return (f"browser_search ({eng}) returned no usable results page (got: {txt[:200]!r}). "
                    "Try another engine or fall back to web_search.")
        if blocked:
            return (f"BLOCKED: {eng} returned a bot check / CAPTCHA. Do NOT try to solve it. "
                    "Try engine='duckduckgo' or 'bing', or fall back to web_search.")
        out = [f'browser_search ({eng}) for "{query}":', txt[:6000], "\nResult links:"]
        out += [f"- {x or '(no text)'} -> {h}" for h, x in ext[:25]]
        return "\n".join(out)

    @tool
    def browser_open(url: str) -> str:
        """Open a URL in the browser (JavaScript is executed) and return the rendered main text and links.

        Use it for pages that need JavaScript, dynamic tables, or sites that block plain HTTP clients.

        Args:
            url: Absolute http(s) URL.
        """
        t0 = time.time()
        msg = _charge("browser_open", url=url)
        if msg:
            return msg

        def go(page):
            page.goto(url, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            u, t, txt = _page_text(page, s.browser_page_chars)
            return u, t, txt, _links(page)
        try:
            u, t, txt, links = session.run(go)
        except Exception as e:
            ctx.log_event(kind="tool", tool="browser_open", agent=agent_id, outcome="error", url=url,
                          error=str(e)[:300], latency=round(time.time() - t0, 2))
            return f"browser_open failed: {e}"
        return _render("browser_open", t0, u, t, txt, links, via="browser_open")

    @tool
    def browser_click(link_text: str) -> str:
        """Click a link or button on the current page by its visible text, then return the new page content.

        Args:
            link_text: Visible text of the link/button (partial match is fine).
        """
        t0 = time.time()
        msg = _charge("browser_click", link_text=link_text)
        if msg:
            return msg

        def go(page):
            loc = page.get_by_role("link", name=link_text)
            if loc.count() == 0:
                loc = page.get_by_text(link_text, exact=False)
            loc.first.click(timeout=10000)
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            u, t, txt = _page_text(page, s.browser_page_chars)
            return u, t, txt, _links(page)
        try:
            u, t, txt, links = session.run(go)
        except Exception as e:
            ctx.log_event(kind="tool", tool="browser_click", agent=agent_id, outcome="error",
                          link_text=link_text, error=str(e)[:300], latency=round(time.time() - t0, 2))
            return f"browser_click failed: {e}"
        return _render("browser_click", t0, u, t, txt, links, via="browser_click")

    @tool
    def browser_read(offset: int = 0) -> str:
        """Read more of the current page's text, starting at a character offset.

        Args:
            offset: Character offset (from the previous output's truncation note).
        """
        msg = _charge("browser_read", offset=offset)
        if msg:
            return msg
        txt = state.get("text", "")
        chunk = txt[offset: offset + s.browser_page_chars]
        if state.get("url"):
            ctx.register_source(state["url"], via="browser_read", fetched=True, evidence=chunk)
        ctx.log_event(kind="tool", tool="browser_read", agent=agent_id, outcome="ok", url=state.get("url"),
                      bytes=len(chunk))
        more = "" if offset + len(chunk) >= len(txt) else \
            f"\n... [truncated; call browser_read(offset={offset + s.browser_page_chars})]"
        return f"[{state.get('sid')}] {state.get('url')} (chars {offset}-{offset + len(chunk)} of {len(txt)})\n{chunk}{more}"

    return [browser_search, browser_open, browser_click, browser_read]
