"""Offline end-to-end test of the orchestration with a scripted fake model and mocked web search.
Verifies: supervisor -> parallel researchers -> tools (search/fetch scope) -> compression -> report -> artifacts.
Run: python -m tests.test_pipeline_offline   (no AWS credentials needed)
"""
import json
import os
import tempfile
from unittest import mock

from strands.models.model import Model

from deep_research import pipeline
from deep_research.config import RunRequest, Settings
from deep_research.tools import search


def _ev_text(t):
    return [{"messageStart": {"role": "assistant"}}, {"contentBlockDelta": {"delta": {"text": t}}},
            {"contentBlockStop": {}}, {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}, "metrics": {"latencyMs": 1}}}]


def _ev_tool(name, inp, tid):
    return [{"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"start": {"toolUse": {"name": name, "toolUseId": tid}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(inp)}}}},
            {"contentBlockStop": {}}, {"messageStop": {"stopReason": "tool_use"}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}, "metrics": {"latencyMs": 1}}}]


class FakeModel(Model):
    def __init__(self, *a, **k): self.cfg = {}
    def get_config(self): return self.cfg
    def update_config(self, **k): self.cfg.update(k)
    async def structured_output(self, *a, **k): raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kw):
        names = [t["name"] for t in (tool_specs or [])]
        n_tool_results = sum(1 for m in messages for b in m["content"] if "toolResult" in b)
        if "conduct_research" in names:            # supervisor: 2 parallel researchers, then stop
            evs = _ev_text("coverage note: all answered") if n_tool_results else (
                _ev_tool("conduct_research", {"subtopic": "sub A"}, "t1")[:4] +
                _ev_tool("conduct_research", {"subtopic": "sub B"}, "t2")[1:])
        elif "web_search" in names:                # researcher
            script = [("web_search", {"query": "q1"}),
                      ("fetch_url", {"url": "https://example.com/"}),
                      ("fetch_url", {"url": "https://iana.org/"})]
            if n_tool_results < len(script) and script[n_tool_results][0] in names:
                name, inp = script[n_tool_results]
                evs = _ev_tool(name, inp, f"r{n_tool_results}")
            else:
                evs = _ev_text("Findings: fact one [S1]. fact two [S2].")
        elif "findings document" in messages[-1]["content"][0].get("text", ""):
            evs = _ev_text("Compressed: fact one [S1]. Gaps: none")
        elif "final research report" in messages[-1]["content"][0].get("text", ""):
            evs = _ev_text("# Report\n- fact one [S1]\n- fact two [S2] [S99]")
        else:
            evs = _ev_text("Research brief: sub-questions ...")
        for e in evs:
            yield e


def main():
    tmp = tempfile.mkdtemp()
    os.environ["LOCAL_RESULTS_DIR"] = tmp
    fake_results = [{"url": "https://example.com/", "title": "Example", "published": "2026-01-01", "text": "snippet"}]
    for mode in ["S1", "S2", "S3"]:
        with mock.patch.object(pipeline, "_model", lambda *a, **k: FakeModel()), \
             mock.patch.object(search, "raw_web_search", lambda *a, **k: fake_results):
            summ = pipeline.run_research(RunRequest(topic="t", mode=mode, depth="quick",
                                                    focus_questions=["q?"]), Settings())
        assert summ["status"] == "completed", summ
        print(mode, summ["tool_outcomes"], summ["citation_stats"]["invalid_citation_ids"],
              summ["token_usage_total"]["totalTokens"])
        files = sorted(os.listdir(os.path.join(tmp, summ["run_id"])))
        assert "summary.json" in files and "report.md" in files, files
    print("OK, artifacts:", files)

    # --- run caps: research stops early, the run still completes with a report -------------------
    def capped(max_cost, price_input):
        s = Settings()
        s.price_input = price_input          # make each fake call "expensive" so the caps trigger
        with mock.patch.object(pipeline, "_model", lambda *a, **k: FakeModel()), \
             mock.patch.object(search, "raw_web_search", lambda *a, **k: fake_results):
            return pipeline.run_research(RunRequest(topic="t", mode="S3", depth="quick", max_cost_usd=max_cost), s)

    # soft limit already reached after the brief -> no researcher is hired
    summ = capped(max_cost=0.5, price_input=50_000.0)   # 10 input tokens = $0.5 per call
    ev = json.load(open(os.path.join(tmp, summ["run_id"], "events.json")))
    assert summ["status"] == "completed" and summ["limits_hit"].get("cost"), summ
    assert not any(e.get("kind") == "researcher_start" for e in ev), "no researcher should start"
    assert os.path.exists(os.path.join(tmp, summ["run_id"], "report.md"))
    print("cap(soft at start):", summ["limits_hit"], summ["est_cost_usd"])

    # cap reached mid-research -> tools refuse, further researcher/supervisor model calls are cancelled
    summ = capped(max_cost=2.0, price_input=20_000.0)   # $0.2 per call
    ev = json.load(open(os.path.join(tmp, summ["run_id"], "events.json")))
    kinds = {e.get("kind") for e in ev}
    outcomes = summ["tool_outcomes"]
    assert summ["status"] == "completed" and summ["limits_hit"], summ
    assert "researcher_start" in kinds, kinds
    print("cap(mid-run):", summ["limits_hit"], outcomes,
          "cancelled_calls=", sum(e.get("kind") == "model_call_cancelled" for e in ev), summ["est_cost_usd"])

    # payload cannot exceed the runtime ceiling
    s = Settings(); s.max_cost_ceiling_usd = 5
    assert RunRequest(topic="t", depth="deep", max_cost_usd=100).caps(s)[0] == 5
    assert RunRequest(topic="t", depth="quick").caps(s) == (3.0, 20.0)
    print("caps OK")


if __name__ == "__main__":
    main()
