"""Run the topic x mode x repeat matrix against the deployed AgentCore Runtime and download results.

  python eval/run_matrix.py                         # full matrix from topics.yaml
  python eval/run_matrix.py --topics dc-infra --modes S1 S3 --repeats 1 --depth quick   # pilot

Each run gets its own runtimeSessionId (own microVM). Runs are started asynchronously and the
harness polls S3 for summary.json, then downloads all artifacts to eval/results/<topic>/<mode>/rep<k>/.
Mode order is interleaved per repeat so time-of-day / index-freshness effects spread evenly.

Spending limits (estimates from list prices, see est_cost_usd in each summary.json):
  --max-cost-per-run  per-run cap sent in the payload (default by depth: quick 3, standard 8, deep 20 USD)
  --budget-usd        total budget for the whole experiment (default 200 USD). Runs already in
                      eval/results count toward it, so a resumed matrix stays inside the same budget.
                      A new run starts only if  spent + caps of runs in flight + its own cap <= budget.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
import uuid

import boto3
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEPTH_COST_CAP = {"quick": 3.0, "standard": 8.0, "deep": 20.0}   # mirrors agent/deep_research/config.py
ARTIFACTS = ["summary.json", "report.md", "sources.json", "events.json", "findings.json", "brief.md",
             "request.json", "supervisor_note.md", "error.txt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=os.path.join(HERE, "..", "infra", "outputs.json"))
    ap.add_argument("--topics", nargs="*")
    ap.add_argument("--modes", nargs="*")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--depth")
    ap.add_argument("--max-in-flight", type=int, default=8)
    ap.add_argument("--timeout-min", type=int, default=90)
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument("--budget-usd", type=float, default=200.0, help="total experiment budget (USD)")
    ap.add_argument("--max-cost-per-run", type=float, default=None, help="per-run cap (USD); default by depth")
    args = ap.parse_args()

    out = json.load(open(args.outputs))
    cfg = yaml.safe_load(open(os.path.join(HERE, "topics.yaml")))
    d = cfg["defaults"]
    topics = cfg["topics"] + (cfg.get("optional_topics", []) if args.include_optional else [])
    if args.topics:
        topics = [t for t in topics if t["id"] in args.topics]
    modes = args.modes or d["modes"]
    reps = args.repeats or d["repeats"]

    sess = boto3.Session(region_name=out["region"])
    rt = sess.client("bedrock-agentcore")
    s3 = sess.client("s3")

    def run_cost(summary_path: str) -> float:
        try:
            return float(json.load(open(summary_path)).get("est_cost_usd", {}).get("total") or 0)
        except Exception:
            return 0.0

    jobs = []
    spent = sum(run_cost(p) for p in glob.glob(os.path.join(HERE, "results", "*", "*", "rep*", "summary.json")))
    for rep in range(1, reps + 1):
        for t in topics:
            order = modes[:]
            random.Random(f"{t['id']}-{rep}").shuffle(order)
            for m in order:
                label = f"{t['id']}/{m}/rep{rep}"
                local = os.path.join(HERE, "results", t["id"], m, f"rep{rep}")
                if os.path.exists(os.path.join(local, "summary.json")):
                    print("skip (done):", label)
                    continue
                depth = args.depth or t.get("depth", d["depth"])
                cap = args.max_cost_per_run or DEPTH_COST_CAP[depth]
                jobs.append({"label": label, "local": local, "cap": cap, "payload": {
                    "action": "start", "topic": t["topic"].strip(), "mode": m,
                    "depth": depth, "max_cost_usd": cap,
                    "language": t.get("language", d["language"]), "audience": t.get("audience", d["audience"]),
                    "focus_questions": t.get("focus_questions", []), "published_after": t.get("published_after"),
                    "run_label": label}})

    print(f"budget ${args.budget_usd:.0f}; already spent ${spent:.2f} on finished runs; {len(jobs)} runs to start")
    pending, running = list(jobs), {}
    stopped = False
    while pending or running:
        while pending and len(running) < args.max_in_flight and not stopped:
            j = pending[0]
            reserved = sum(r["cap"] for r in running.values())
            if spent + reserved + j["cap"] > args.budget_usd:
                if not running:
                    print(f"BUDGET STOP: spent ${spent:.2f}; next run could need ${j['cap']:.2f} "
                          f"> budget ${args.budget_usd:.2f}. {len(pending)} runs not started.")
                    stopped = True
                break
            pending.pop(0)
            sid = f"dr-{uuid.uuid4().hex}"  # >= 33 chars
            resp = rt.invoke_agent_runtime(agentRuntimeArn=out["runtime_arn"], runtimeSessionId=sid,
                                           contentType="application/json", accept="application/json",
                                           payload=json.dumps(j["payload"]).encode())
            body = json.loads(resp["response"].read())
            j.update(run_id=body["run_id"], session=sid, t0=time.time())
            running[j["run_id"]] = j
            print(f"started {j['label']} -> {j['run_id']}")
        time.sleep(30)
        for rid, j in list(running.items()):
            prefix = f"{out['results_prefix']}/{rid}/"
            try:
                s3.head_object(Bucket=out["bucket"], Key=prefix + "summary.json")
            except Exception:
                if time.time() - j["t0"] > args.timeout_min * 60:
                    spent += j["cap"]  # unknown cost: assume it used its whole cap
                    print("TIMEOUT", j["label"], f"(counted ${j['cap']:.2f})")
                    running.pop(rid)
                continue
            os.makedirs(j["local"], exist_ok=True)
            for name in ARTIFACTS:
                try:
                    s3.download_file(out["bucket"], prefix + name, os.path.join(j["local"], name))
                except Exception:
                    pass
            summ = json.load(open(os.path.join(j["local"], "summary.json")))
            cost = float(summ.get("est_cost_usd", {}).get("total") or 0)
            spent += cost
            lim = f" LIMIT {summ['limits_hit']}" if summ.get("limits_hit") else ""
            print(f"done {j['label']}: {summ['status']} {summ['elapsed_s']}s ${cost:.2f}{lim} "
                  f"| total ${spent:.2f} / ${args.budget_usd:.0f}")
            running.pop(rid)
        if stopped and not running:
            break
    print(f"finished: est. spend ${spent:.2f} of ${args.budget_usd:.0f}")


if __name__ == "__main__":
    main()
