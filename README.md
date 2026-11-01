# Deep Research Agent on AgentCore — retrieval-setting experiment (S1–S4)

One deep-research agent, four retrieval settings, identical everything else, so you can measure what
page-level access actually buys you.

| Mode | Tools given to researcher sub-agents | Egress from your runtime |
|---|---|---|
| **S1** `search_index_only` | AgentCore Web Search (managed connector on AgentCore Gateway) | none (queries served inside AWS) |
| **S2** `search_plus_fetch_scoped` | Web Search + `fetch_url` restricted to URLs returned by Web Search in this run | only to search-result URLs |
| **S3** `search_plus_fetch_open` | Web Search + `fetch_url` on any public URL (follow links, PDFs) | any public URL |
| **S4** `search_plus_browser` | Web Search + AgentCore Browser (`browser_search` via Bing/DDG/Google SERP, `browser_open`, `browser_click`, `browser_read`) | from the AWS-managed browser |

## Why this base

Surveyed (Oct 2026): LangChain **open_deep_research** (≈12.7k★, MIT, #6 on DeepResearch Bench at RACE 0.4344 —
**archived Aug 2026**, LangChain moved to `deepagents`), **GPT Researcher** (≈27k★, Apache-2.0, fixed
planner→scrape→publish pipeline that always scrapes pages, so it cannot express S1), **Tongyi DeepResearch**
(≈20k★, needs its own 30B model), **aws-samples/sample-agentcore-deep-research** (Strands + AgentCore + CDK,
single agent, Lambda data-source tools, no fetch/browser path).

This repo re-implements the benchmarked open_deep_research architecture
(brief → supervisor → parallel researchers → per-researcher compression → report) on **Strands Agents +
AgentCore Runtime**, because the experiment needs (a) one agent where only the tool set changes, (b) native
AgentCore Web Search / Browser, and (c) per-call telemetry. Nothing in the pipeline is mode-specific except
the tool list and a 2–3 line tool description.

## Layout

```
agent/
  app.py                      AgentCore Runtime entrypoint (async start / status / sync run)
  cli.py                      run locally with your AWS creds
  deep_research/
    config.py                 modes, depth budgets, env settings, RunRequest (dynamic input)
    pipeline.py               brief -> supervisor -> researchers -> compress -> report
    prompts.py                prompts (only TOOL_GUIDANCE differs per mode)
    run_context.py            source registry ([S12] IDs + evidence seen), telemetry, budgets, S3 store
    tools/search.py           AgentCore Web Search via Gateway (MCP, SigV4)
    tools/fetch.py            fetch_url (HTML via trafilatura, PDF via pypdf, robots.txt, SSRF guard, S2 allowlist)
    tools/browser.py          AgentCore Browser over CDP with Playwright (dedicated thread per researcher)
  tests/                      offline pipeline test (fake model) + browser tools vs local Chromium
  Dockerfile, requirements.txt
infra/
  build_push.sh               build arm64 image -> ECR
  setup.py                    bucket, gateway (AWS_IAM) + web-search target, roles, runtime -> outputs.json
eval/
  topics.yaml                 4 topics (+1 optional) with must-answer probes and pre-registered expected spread
  run_matrix.py               topic x mode x repeat runner (one session per run), downloads artifacts
  judge.py                    rubric, probes, citation grounding, process metrics -> scores/summary.md
```

## Deploy

```bash
# region must have Web Search: us-east-1, us-east-2, us-west-2, eu-west-1, ap-southeast-2, ap-northeast-1
IMAGE=$(./infra/build_push.sh us-west-2 | tail -1)
python infra/setup.py --region us-west-2 --image "$IMAGE" --model-id global.anthropic.claude-sonnet-5-5
```

Runtime uses `PUBLIC` network mode so S2/S3 can fetch. If you move it into a VPC, S2/S3 need NAT or an
egress proxy; S1 and S4 do not need runtime egress (Web Search is reached through Gateway, the browser runs
in AWS-managed infrastructure).

## Invoke (dynamic input)

```python
payload = {
  "action": "start",                 # start | status | run (sync, quick only) | modes
  "topic": "…any research question…",
  "mode": "S3",                      # S1 | S2 | S3 | S4
  "depth": "standard",               # quick | standard | deep  (same budget for every mode)
  "language": "zh-TW",
  "audience": "Taiwan enterprise architects",
  "focus_questions": ["must-answer 1", "must-answer 2"],
  "published_after": "2025-01-01",   # -> Web Search publishedDateFilter (connector >= 1.2.0)
  "include_domains": [], "exclude_domains": [],
  "report_instructions": "",
  "max_cost_usd": 8,                 # optional per-run caps (defaults by depth, see Spending limits)
  "max_minutes": 45
}
rt = boto3.client("bedrock-agentcore", region_name="us-west-2")
r = rt.invoke_agent_runtime(agentRuntimeArn=ARN, runtimeSessionId="dr-"+uuid.uuid4().hex,
                            payload=json.dumps(payload).encode())
print(json.loads(r["response"].read()))   # {"run_id": ..., "results": "s3://…/<run_id>/"}
```

Runs go async because deep research regularly exceeds the 15-minute synchronous request limit; the entrypoint
registers an async task so the session stays `HealthyBusy`. Artifacts per run:
`report.md, sources.json (every source + the exact text the agent saw), events.json (every tool call),
findings.json, brief.md, summary.json (written last)`.

## Run the comparison

```bash
python eval/run_matrix.py --topics dc-infra --modes S1 S3 --repeats 1 --depth quick   # pilot first
python eval/run_matrix.py                    # 4 topics x 4 modes x 3 repeats = 48 runs
python eval/judge.py --refetch               # scores/summary.md
```

Put your verified answers to the probe questions in `eval/ground_truth.yaml`
(`{dc-infra: {1: "...", 2: "..."}, ...}`) to get `probe_correct_rate`.

## Cost

All stages use Claude Sonnet 5.5 (`global.anthropic.claude-sonnet-5-5`). Bedrock prompt caching is **on by default**
(`PROMPT_CACHE=true`): each agent-loop turn re-reads the growing conversation at the cache-read rate, which cuts
model cost by roughly 60% and is applied identically to every mode. Set `PROMPT_CACHE=false` on the runtime to
measure uncached cost.

Every `summary.json` reports `token_usage_total` (incl. cache read/write tokens), `cache_hit_ratio` and
`est_cost_usd` (model + Web Search, from list prices in `config.py`; override with `PRICE_*` env vars).
Rough budget at `standard` depth with caching: S1 ≈ $2, S2–S4 ≈ $2.5–3.5 per run; the 48-run matrix ≈ $130
plus ≈ $25 for `judge.py --refetch`.

### Spending limits

| Layer | Setting | Default | What happens |
|---|---|---|---|
| Per run | payload `max_cost_usd`, `max_minutes` | by depth: quick $3 / 20 min, standard $8 / 45 min, deep $20 / 90 min | at 80%: no new tool calls or researchers; at 100%: further researcher/supervisor model calls are cancelled. Compression and the final report always run, so a capped run still returns a report (actual spend can exceed the cap by roughly the cost of the report). |
| Runtime ceiling | env `MAX_COST_PER_RUN_USD`, `MAX_MINUTES_PER_RUN` | $30, 120 min | a payload can never ask for more than this |
| Whole experiment | `run_matrix.py --budget-usd` | $200 | a run starts only if spent + caps of runs in flight + its own cap ≤ budget; finished runs in `eval/results` count, so resuming stays inside the same budget |

Caps are a safety net, not the experimental budget: `summary.json` → `limits_hit` and the judge's `limit_hit`
column flag capped runs. If one mode hits the cap often, raise it, otherwise the cap decides the comparison.
All figures are estimates from list prices; for a billing-side backstop add an AWS Budgets alert
(billing data lags by hours, so it cannot stop a run in progress).

## Fairness controls built in

- Same model, prompts, budgets (`max_tool_calls` per researcher, researchers, parallelism) for every mode.
- `fetch_url` output is identical in S2 and S3 (including outbound links); only the allowlist differs.
  Rejected S2 attempts do not consume budget and are logged as `blocked_scope` ("wanted to go deeper").
- Mode order shuffled per repeat; one microVM per run.
- Judge is blind: the auto-generated Sources list (which shows read vs snippet-only) is stripped.
- Grounding is checked against the evidence the agent actually saw (`sources.json`), separating
  "the tool gave bad evidence" from "the model wrote beyond its evidence".

## Tests (no AWS needed)

```bash
cd agent && python -m tests.test_pipeline_offline   # orchestration, S2 allowlist, artifacts
python -m tests.test_browser_local                  # browser tools vs local Chromium (needs `playwright install chromium`)
```
