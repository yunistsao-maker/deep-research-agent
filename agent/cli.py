"""Run one research job locally (same code path as the runtime), using your local AWS credentials.

  export GATEWAY_URL=https://<gw-id>.gateway.bedrock-agentcore.us-west-2.amazonaws.com/mcp
  python cli.py --mode S2 --budget-mode client --depth quick --topic "..." [--focus "question 1" --focus "question 2"]
"""
import argparse
import json
import logging

from deep_research.config import RunRequest, Settings
from deep_research.pipeline import run_research

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--mode", default="S1", choices=["S1", "S2", "S3", "S4"])
    ap.add_argument("--depth", default="quick", choices=["quick", "standard", "deep"])
    ap.add_argument("--budget-mode", default="none", choices=["none", "native", "client"])
    ap.add_argument("--task-budget-tokens", type=int, default=None)
    ap.add_argument("--language", default="zh-TW")
    ap.add_argument("--focus", action="append", default=[], help="must-answer question (repeatable)")
    ap.add_argument("--published-after", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    req = RunRequest(topic=args.topic, mode=args.mode, depth=args.depth, language=args.language,
                     focus_questions=args.focus, published_after=args.published_after,
                     budget_mode=args.budget_mode, task_budget_tokens=args.task_budget_tokens)
    print(json.dumps(run_research(req, Settings()), ensure_ascii=False, indent=2))
