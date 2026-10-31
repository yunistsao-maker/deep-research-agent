"""Run the topic x mode x repeat matrix against the deployed AgentCore Runtime and download results.

  python eval/run_matrix.py                         # full matrix from topics.yaml
  python eval/run_matrix.py --topics dc-infra --modes S1 S3 --repeats 1 --depth quick   # pilot

Each run gets its own runtimeSessionId (own microVM). Runs are started asynchronously and the
harness polls S3 for summary.json, then downloads all artifacts to eval/results/<topic>/<mode>/rep<k>/.
Mode order is interleaved per repeat so time-of-day / index-freshness effects spread evenly.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid

import boto3
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
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

    jobs = []
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
                jobs.append({"label": label, "local": local, "payload": {
                    "action": "start", "topic": t["topic"].strip(), "mode": m,
                    "depth": args.depth or t.get("depth", d["depth"]),
                    "language": t.get("language", d["language"]), "audience": t.get("audience", d["audience"]),
                    "focus_questions": t.get("focus_questions", []), "published_after": t.get("published_after"),
                    "run_label": label}})

    pending, running = list(jobs), {}
    while pending or running:
        while pending and len(running) < args.max_in_flight:
            j = pending.pop(0)
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
                    print("TIMEOUT", j["label"])
                    running.pop(rid)
                continue
            os.makedirs(j["local"], exist_ok=True)
            for name in ARTIFACTS:
                try:
                    s3.download_file(out["bucket"], prefix + name, os.path.join(j["local"], name))
                except Exception:
                    pass
            summ = json.load(open(os.path.join(j["local"], "summary.json")))
            print(f"done {j['label']}: {summ['status']} {summ['elapsed_s']}s "
                  f"tools={summ['tool_calls']} tokens={summ['token_usage_total'].get('totalTokens')}")
            running.pop(rid)


if __name__ == "__main__":
    main()
