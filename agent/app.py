"""AgentCore Runtime entrypoint.

Invoke payloads:
  {"action": "start", "topic": "...", "mode": "S1|S2|S3|S4", "depth": "quick|standard|deep", ...}
      -> starts the run in the background (deep research often exceeds the 15-minute synchronous
         request limit), returns {"run_id", "status": "started"}. Results land in
         s3://$RESULTS_BUCKET/$RESULTS_PREFIX/<run_id>/ ; summary.json is written last.
  {"action": "status", "run_id": "..."}   (same runtimeSessionId as "start")
  {"action": "run", ...}                  synchronous; only for depth=quick smoke tests
  {"action": "modes"}                     describe the four settings

Use one runtimeSessionId per run so every run gets its own microVM (isolation + parallel runs).
"""
from __future__ import annotations

import logging
import threading

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from deep_research.config import MODES, DEPTHS, RunRequest, Settings
from deep_research.pipeline import run_research
from deep_research.run_context import RunContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = BedrockAgentCoreApp()
SETTINGS = Settings()
RUNS: dict[str, RunContext] = {}


def _background(ctx: RunContext, task_id: int) -> None:
    try:
        run_research(ctx.req, SETTINGS, ctx)
    finally:
        app.complete_async_task(task_id)  # runtime ping goes back to Healthy -> session can idle out


@app.entrypoint
def invoke(payload: dict, context=None) -> dict:
    action = (payload or {}).get("action", "start")
    if action == "modes":
        return {"modes": MODES, "depths": DEPTHS, "model_id": SETTINGS.model_id}
    if action == "status":
        ctx = RUNS.get(payload.get("run_id", ""))
        if not ctx:
            return {"error": "unknown run_id in this session; read summary.json from S3 instead"}
        return ctx.summary()
    req = RunRequest.from_payload(payload)
    if action == "run":
        return run_research(req, SETTINGS)
    ctx = RunContext(req, SETTINGS)
    RUNS[ctx.run_id] = ctx
    # registers an async task -> /ping reports HealthyBusy so the session is not reclaimed mid-run
    task_id = app.add_async_task("deep_research", {"run_id": ctx.run_id, "mode": req.mode})
    threading.Thread(target=_background, args=(ctx, task_id), daemon=True).start()
    return {"run_id": ctx.run_id, "status": "started", "mode": req.mode, "depth": req.depth,
            "results": f"s3://{SETTINGS.results_bucket}/{SETTINGS.results_prefix}/{ctx.run_id}/"
            if SETTINGS.results_bucket else f"{SETTINGS.local_results_dir}/{ctx.run_id}/"}


if __name__ == "__main__":
    app.run()
