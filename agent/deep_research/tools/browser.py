"""AgentCore Browser tools.

S4 (search only): browser_search runs the query on a public search engine in a real browser and returns
     the organic results (title, URL, snippet) as citable sources. No page is opened.
S5 (full browser, optional): browser_search + browser_open / browser_click / browser_read to open
     JS-rendered pages, click through and read page text.

strands_tools.browser.AgentCoreBrowser is deprecated, so we drive Playwright directly over CDP using
bedrock-agentcore's BrowserClient, as its deprecation notice recommends.

Playwright's sync API must stay on one thread, while Strands runs sync tools in worker threads,
so each researcher gets a BrowserSession that owns a dedicated thread and executes commands from a queue.
"""
from __future__ import annotations

import base64
import queue
import threading
import time
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, quote_plus, unquote, urlsplit

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


# Organic results on Bing / DuckDuckGo (html) / Google result pages: [href, title, snippet]
SERP_JS = """() => {
  const out = [];
  const push = (a, t, p) => { if (a && a.href) out.push([a.href, (t || '').trim(), (p || '').trim()]); };
  document.querySelectorAll('li.b_algo').forEach(li => {
    const a = li.querySelector('h2 a');
    const p = li.querySelector('.b_caption p, p.b_lineclamp2, p.b_lineclamp3, p.b_lineclamp4, .b_snippet, p');
    push(a, a && a.innerText, p && p.innerText);
  });
  if (!out.length) document.querySelectorAll('.result').forEach(r => {
    const a = r.querySelector('a.result__a');
    const p = r.querySelector('.result__snippet');
    push(a, a && a.innerText, p && p.innerText);
  });
  if (!out.length) document.querySelectorAll('div.g').forEach(g => {
    const h = g.querySelector('h3'); const a = h && h.closest('a');
    const p = g.querySelector('div.VwiC3b, [data-sncf], span.st');
    push(a, h && h.innerText, p && p.innerText);
  });
  return out.slice(0, 15);
}"""


def decode_result_url(href: str) -> str:
    """Unwrap search-engine redirect links (Bing /ck/a?u=a1<base64>, DuckDuckGo /l/?uddg=, Google /url?q=)."""
    try:
        p = urlsplit(href)
        q = parse_qs(p.query)
        host = p.netloc.lower()
        if "bing.com" in host and "u" in q and q["u"][0].startswith("a1"):
            b = q["u"][0][2:]
            return base64.urlsafe_b64decode(b + "=" * (-len(b) % 4)).decode("utf-8", "replace")
        if "duckduckgo.com" in host and "uddg" in q:
            return unquote(q["uddg"][0])
        if p.path == "/url" and ("q" in q or "url" in q):          # Google redirect (any Google domain)
            target = (q.get("q") or q.get("url"))[0]
            if target.startswith(("http://", "https://")):
                return target
    except Exception:
        pass
    return href


_ENGINE_HOSTS = ("bing.com", "duckduckgo.com", "google.", "microsoft.com", "msn.com")


def parse_serp_results(raw: list) -> list[dict]:
    seen, out = set(), []
    for href, title, snippet in raw or []:
        url = decode_result_url(href)
        host = urlsplit(url).netloc.lower()
        if not url.startswith(("http://", "https://")) or any(h in host for h in _ENGINE_HOSTS):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append({"url": url, "title": " ".join(title.split())[:200], "snippet": " ".join(snippet.split())[:600]})
    return out



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


def make_browser_tools(ctx: RunContext, agent_id: str, session: BrowserSession, search_only: bool = False):
    """search_only=True (mode S4): only browser_search; the agent sees result snippets but cannot open pages."""
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
        """Search the web through a real browser on a public search engine (Bing by default).

        Returns the organic results the engine shows: a source ID like [S12], title, URL and the engine's
        snippet for each. Cite facts with the source ID. Use it to find sources web_search missed
        (local-language, niche or very recent pages).

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
            return page.title(), page.inner_text("body")[:3000], page.evaluate(SERP_JS)
        try:
            title, body, raw = session.run(go)
        except Exception as e:
            ctx.log_event(kind="tool", tool="browser_search", agent=agent_id, outcome="error", query=query,
                          error=str(e)[:300], latency=round(time.time() - t0, 2))
            return f"browser_search failed: {e}"
        blocked = any(m in (title + " " + body).lower() for m in BLOCK_MARKERS)
        results = parse_serp_results(raw)
        outcome = "blocked_bot" if blocked and not results else ("ok" if results else "empty")
        ctx.log_event(kind="tool", tool="browser_search", agent=agent_id, outcome=outcome, query=query,
                      engine=eng, n_results=len(results), urls=[r["url"] for r in results],
                      bytes=sum(len(r["snippet"]) for r in results), latency=round(time.time() - t0, 2))
        if outcome == "blocked_bot":
            return (f"BLOCKED: {eng} returned a bot check / CAPTCHA. Do NOT try to solve it. "
                    "Try engine='bing' or 'duckduckgo', or use web_search.")
        if outcome == "empty":
            return (f"browser_search ({eng}) found no results for \"{query}\". Rephrase, try another engine, "
                    "or use web_search.")
        ctx.add_search_results([r["url"] for r in results])
        lines = [f'browser_search ({eng}) results for "{query}" ({len(results)} results):']
        for r in results:
            src = ctx.register_source(r["url"], r["title"], via="browser_search", evidence=r["snippet"])
            lines.append(f'\n[{src.sid}] {r["title"] or "(no title)"}\nURL: {r["url"]}\n'
                         f'Snippet: {r["snippet"] or "(no snippet shown)"}')
        return "\n".join(lines)

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

    if search_only:
        return [browser_search]
    return [browser_search, browser_open, browser_click, browser_read]
