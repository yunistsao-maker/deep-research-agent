"""Exercise the S4 browser tools against a LOCAL Chromium (stands in for the AgentCore Browser CDP endpoint).
Run: python -m tests.test_browser_local
"""
from unittest import mock

from deep_research.config import RunRequest, Settings
from deep_research.run_context import RunContext
from deep_research.tools import browser as B


class FakeClient:
    def __init__(self, region): pass
    def start(self, **kw): return "local-session"
    def generate_ws_headers(self): return "local", {}
    def stop(self): return True


def main():
    from playwright.sync_api import BrowserType
    real_launch = BrowserType.launch
    def fake_connect(self, ws, headers=None):
        return real_launch(self, headless=True)
    with mock.patch.object(B, "BrowserClient", FakeClient), \
         mock.patch.object(BrowserType, "connect_over_cdp", fake_connect):
        ctx = RunContext(RunRequest(topic="t", mode="S4", depth="quick"), Settings())
        sess = B.BrowserSession("us-west-2", "aws.browser.v1")
        search_t, open_t, click_t, read_t = B.make_browser_tools(ctx, "R1", sess)
        print(open_t(url="https://example.com")[:300])
        print("---"); print(click_t(link_text="Learn more")[:300])
        print("---"); print(read_t(offset=100)[:200])
        print("---"); print(search_t(query="AgentCore web search", engine="duckduckgo")[:400])
        print("session seconds:", sess.close())
        print(ctx.summary()["tool_outcomes"])


if __name__ == "__main__":
    main()
