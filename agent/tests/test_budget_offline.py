"""Offline test of the budget_mode factor (none / native / client) with fake models.

- client: footers with the remaining budget appear on tool results; an obedient model stops when told,
          counted tokens stay near the budget; tools refuse once it is exhausted.
- native: no footers (the countdown is server-side); a model that ignores the budget is stopped by the
          1.5x backstop.
- none:   no budget, but counted tokens are still measured for comparison.
Run: python -m tests.test_budget_offline   (no AWS credentials needed)
"""
import json
import os
import tempfile
from unittest import mock

from strands.models.model import Model

from deep_research import config, pipeline
from deep_research.config import RunRequest, Settings
from deep_research.tools import search


def _usage_events(inp):
    return {"metadata": {"usage": {"inputTokens": inp, "outputTokens": 50, "totalTokens": inp + 50},
                         "metrics": {"latencyMs": 1}}}


def _text(t, inp):
    return [{"messageStart": {"role": "assistant"}}, {"contentBlockDelta": {"delta": {"text": t}}},
            {"contentBlockStop": {}}, {"messageStop": {"stopReason": "end_turn"}}, _usage_events(inp)]


def _tool(name, args, tid, inp):
    return [{"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"start": {"toolUse": {"name": name, "toolUseId": tid}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}},
            {"contentBlockStop": {}}, {"messageStop": {"stopReason": "tool_use"}}, _usage_events(inp)]


def _ctx_tokens(messages, system_prompt):
    n = len(system_prompt or "")
    for m in messages:
        for b in m["content"]:
            n += len(json.dumps(b))
    return n // 4


class FakeModel(Model):
    """obedient=True stops when a tool result says so; False ignores budgets (keeps searching)."""

    def __init__(self, obedient=True):
        self.cfg, self.obedient, self.seen_texts = {}, obedient, []

    def get_config(self): return self.cfg
    def update_config(self, **k): self.cfg.update(k)
    async def structured_output(self, *a, **k): raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, system_prompt_content=None, **kw):
        system_prompt = system_prompt or "".join(b.get("text", "") for b in (system_prompt_content or []))
        inp = _ctx_tokens(messages, system_prompt)
        names = [t["name"] for t in (tool_specs or [])]
        results = [b["toolResult"] for m in messages for b in m["content"] if "toolResult" in b]
        last = "".join(c.get("text", "") for c in (results[-1]["content"] if results else []))
        self.seen_texts.append(last)
        if "conduct_research" in names:
            evs = _text("coverage note", inp) if results else _tool("conduct_research", {"subtopic": "A"}, "s1", inp)
        elif "web_search" in names:
            told_to_stop = any(k in last for k in ("do not call more tools", "TOKEN_BUDGET_EXHAUSTED",
                                                   "BUDGET_EXHAUSTED", "RUN_LIMIT"))
            if len(results) >= 40 or (self.obedient and told_to_stop):
                evs = _text("Findings: fact [S1].", inp)
            else:
                evs = _tool("web_search", {"query": f"q{len(results)}"}, f"r{len(results)}", inp)
        else:
            evs = _text("text [S1]", inp)
        for e in evs:
            yield e


def run(mode_budget, obedient, budget=6000):
    researcher_models = []

    def make_model(*a, **k):
        return FakeModel(obedient)

    def make_native(*a, **k):
        m = FakeModel(obedient)
        researcher_models.append(m)
        return m

    big = [{"url": f"https://ex.com/{i}", "title": f"T{i}", "published": "2026-01-01", "text": "evidence " * 220}
           for i in range(2)]
    s = Settings()
    s.max_cost_ceiling_usd = 1000
    with mock.patch.object(pipeline, "_model", make_model), \
         mock.patch.object(pipeline, "_native_model", make_native), \
         mock.patch.object(config, "TASK_BUDGET_MIN", 1000), \
         mock.patch.object(search, "raw_web_search", lambda *a, **k: big):
        req = RunRequest(topic="t", mode="S1", depth="quick", budget_mode=mode_budget,
                         task_budget_tokens=budget if mode_budget != "none" else None, max_cost_usd=1000)
        summ = pipeline.run_research(req, s)
    events = json.load(open(os.path.join(s.local_results_dir, summ["run_id"], "events.json")))
    return summ, events, researcher_models


def main():
    os.environ["LOCAL_RESULTS_DIR"] = tempfile.mkdtemp()

    summ, ev, _ = run("client", obedient=True)
    rb = summ["researcher_budgets"]["R1"]
    stages = [e["stage"] for e in ev if e.get("kind") == "token_budget"]
    print("client/obedient:", rb, "tool calls:", summ["researcher_tool_calls"], "stages:", stages)
    assert summ["status"] == "completed" and "wrap_up" in stages and "stop" in stages
    assert rb["ratio"] <= 1.15, rb
    assert summ["researcher_tool_calls"]["R1"] < 16, "tool-call backstop should not be what stopped it"

    summ, ev, _ = run("client", obedient=False)
    rb = summ["researcher_budgets"]["R1"]
    outcomes = [e.get("stage") for e in ev if e.get("kind") == "token_budget"]
    print("client/ignores:", rb, "stages:", outcomes)
    assert "exhausted" in outcomes, "tools must refuse once the budget is used up"
    assert rb["ratio"] < 1.4, rb

    summ, ev, models = run("native", obedient=False)
    rb = summ["researcher_budgets"]["R1"]
    stages = [e.get("stage") for e in ev if e.get("kind") == "token_budget"]
    footers = sum("[token budget:" in t for m in models for t in m.seen_texts)
    print("native/ignores:", rb, "stages:", stages, "footers seen:", footers)
    assert footers == 0, "native mode must not add client-side footers"
    assert "overrun_cancel" in stages and rb["ratio"] < 1.8, rb

    summ, ev, _ = run("none", obedient=True)
    rb = summ["researcher_budgets"]["R1"]
    print("none:", rb, "tool calls:", summ["researcher_tool_calls"])
    assert rb["total"] is None and rb["counted"] > 0
    assert summ["researcher_tool_calls"]["R1"] <= config.DEPTHS["quick"]["max_tool_calls"] + 1
    print("budget modes OK")


if __name__ == "__main__":
    main()
