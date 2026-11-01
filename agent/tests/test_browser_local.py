"""Exercise the browser tools against a LOCAL Chromium (stands in for the AgentCore Browser CDP endpoint).

1. S4 search-only: browser_search on saved Bing / DuckDuckGo / Google result pages served from localhost
   (redirect links decoded, results registered as citable sources, no page-opening tools exposed).
2. S5 full browser: open / click / read on example.com.
Run: python -m tests.test_browser_local   (needs `playwright install chromium`)
"""
import base64
import functools
import http.server
import os
import tempfile
import threading
from unittest import mock

from deep_research.config import RunRequest, Settings
from deep_research.run_context import RunContext
from deep_research.tools import browser as B


class FakeClient:
    def __init__(self, region): pass
    def start(self, **kw): return "local-session"
    def generate_ws_headers(self): return "local", {}
    def stop(self): return True


def _b64(u):
    return "a1" + base64.urlsafe_b64encode(u.encode()).decode().rstrip("=")


BING = """<html><head><meta charset='utf-8'><title>q - Search</title></head><body><ol id='b_results'>
<li class='b_ad'><h2><a href='https://ads.example/x'>Ad</a></h2></li>
<li class='b_algo'><h2><a href='https://www.bing.com/ck/a?!&&p=abc&u={u1}&ntb=1'>TSMC Kaohsiung fab EIA</a></h2>
  <div class='b_caption'><p>環評承諾：再生水使用比例… reclaimed water 10% by 2030.</p></div></li>
<li class='b_algo'><h2><a href='https://esg.tsmc.com/report.pdf'>TSMC Sustainability Report</a></h2>
  <div class='b_caption'><p class='b_lineclamp2'>Renewable energy share and water recycling rate.</p></div></li>
<li class='b_algo'><h2><a href='https://www.bing.com/videos/search?q=x'>Videos</a></h2></li>
</ol></body></html>""".format(u1=_b64("https://www.moenv.gov.tw/eia/case/123"))

DDG = """<html><head><meta charset='utf-8'></head><body><div class='result'><a class='result__a' href='//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.ocp.org%2Fspec%3Fv%3D2&rut=1'>OCP spec</a>
<a class='result__snippet'>800 VDC rack power distribution.</a></div></body></html>"""

GOOGLE = """<html><head><meta charset='utf-8'></head><body><div class='g'><a href='/url?q=https://developer.nvidia.com/blog/800v&sa=U'><h3>NVIDIA 800 VDC</h3></a>
<div class='VwiC3b'>Next-gen AI factories at 1 MW per rack.</div></div></body></html>"""

CAPTCHA = "<html><head><title>Captcha</title></head><body>Our systems have detected unusual traffic.</body></html>"


def serp_test(B, FakeClient, fake_connect, BrowserType, real_launch):
    d = tempfile.mkdtemp()
    for name, html in (("bing", BING), ("ddg", DDG), ("google", GOOGLE), ("captcha", CAPTCHA)):
        open(os.path.join(d, f"{name}.html"), "w", encoding="utf-8").write(html)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=d)
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    handler = functools.partial(Quiet, directory=d)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    urls = {"bing": base + "/bing.html?q={q}", "duckduckgo": base + "/ddg.html?q={q}",
            "google": base + "/google.html?q={q}", "captcha": base + "/captcha.html?q={q}"}
    with mock.patch.object(B, "BrowserClient", FakeClient), \
         mock.patch.object(BrowserType, "connect_over_cdp", fake_connect), \
         mock.patch.dict(B.SEARCH_URLS, urls):
        ctx = RunContext(RunRequest(topic="t", mode="S4", depth="quick"), Settings())
        sess = B.BrowserSession("us-west-2", "aws.browser.v1")
        tools = B.make_browser_tools(ctx, "R1", sess, search_only=True)
        assert [t.tool_name for t in tools] == ["browser_search"], "S4 must expose browser_search only"
        out = tools[0](query="tsmc kaohsiung eia", engine="bing"); print(out); print("---")
        assert "URL: https://www.moenv.gov.tw/eia/case/123" in out and "esg.tsmc.com/report.pdf" in out
        assert "ads.example" not in out and "bing.com" not in out
        out = tools[0](query="ocp 800vdc", engine="duckduckgo"); print(out); print("---")
        assert "URL: https://www.ocp.org/spec?v=2" in out
        out = tools[0](query="nvidia 800v", engine="google"); print(out); print("---")
        assert "URL: https://developer.nvidia.com/blog/800v" in out
        out = tools[0](query="x", engine="captcha"); print(out); print("---")
        assert out.startswith("BLOCKED")
        sess.close()
        srcs = ctx.sources_dump()
        assert all(s["first_seen_via"] == "browser_search" and not s["fetched"] for s in srcs), srcs
        assert any("再生水" in "".join(s["evidence"]) for s in srcs), "snippet must be kept as evidence"
        print("S4 search-only OK:", len(srcs), "sources;", ctx.summary()["tool_outcomes"])
    srv.shutdown()


def main():
    from playwright.sync_api import BrowserType
    real_launch = BrowserType.launch
    def fake_connect(self, ws, headers=None):
        return real_launch(self, headless=True)
    serp_test(B, FakeClient, fake_connect, BrowserType, real_launch)
    with mock.patch.object(B, "BrowserClient", FakeClient), \
         mock.patch.object(BrowserType, "connect_over_cdp", fake_connect):
        ctx = RunContext(RunRequest(topic="t", mode="S5", depth="quick"), Settings())
        sess = B.BrowserSession("us-west-2", "aws.browser.v1")
        search_t, open_t, click_t, read_t = B.make_browser_tools(ctx, "R1", sess)
        print(open_t(url="https://example.com")[:300])
        print("---"); print(click_t(link_text="Learn more")[:300])
        print("---"); print(read_t(offset=100)[:200])
        print("session seconds:", sess.close())
        print(ctx.summary()["tool_outcomes"])


if __name__ == "__main__":
    main()
