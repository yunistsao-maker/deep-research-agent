"""Score every run and aggregate by topic x mode.

  python eval/judge.py                    # rubric + probes + citation grounding (vs. evidence the agent saw)
  python eval/judge.py --refetch          # also re-open cited URLs and check support against the live page
  python eval/judge.py --aggregate-only   # recompute tables from cached scores

Outputs: eval/scores/<topic>__<mode>__<rep>.json, eval/scores/runs.csv, eval/scores/summary.md

Metrics
  quality      rubric 1-10 (comprehensiveness, depth, specificity, source_quality, instruction_following,
               readability) from a blind judge (mode labels and the read/snippet tags are stripped)
  probes       per must-answer question: full / partial / none (+ correct / wrong if ground_truth.yaml exists)
  grounding    sampled claim->citation pairs judged against the text the agent actually saw for that source
               (seen_support_rate) and, with --refetch, against the live page (page_support_rate)
  process      tool calls, outcomes (blocked_scope, http_403, blocked_bot ...), sources read, PDFs, tokens, time
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import statistics as st
from collections import defaultdict

import boto3
import yaml
from botocore.config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
RES, OUT = os.path.join(HERE, "results"), os.path.join(HERE, "scores")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL_ID", "global.anthropic.claude-sonnet-5-5")
REGION = os.environ.get("AWS_REGION", "us-west-2")
_br = boto3.client("bedrock-runtime", region_name=REGION, config=Config(read_timeout=600, retries={"max_attempts": 8}))

RUBRIC = """You are an expert reviewer grading a research report. You do not know how it was produced.
Topic: {topic}
Must-answer questions:
{probes}

<report>
{report}
</report>

Score each criterion 1-10 (10 = expert-analyst quality), judging content, not length:
- comprehensiveness: covers the important aspects of the topic
- depth: insight, mechanisms, trade-offs, non-obvious analysis (not just restating facts)
- specificity: concrete figures, dates, versions, names that a domain expert would expect
- source_quality: relies on primary/authoritative sources as far as the citations indicate
- instruction_following: answers every must-answer question explicitly; follows requested structure/language
- readability: clear structure, tables where useful, no filler
For each must-answer question give coverage: "full" (specific answer), "partial" (vague/incomplete), "none",
and copy the report's answer in <= 40 words.
Return ONLY JSON: {{"scores": {{"comprehensiveness": n, "depth": n, "specificity": n, "source_quality": n,
"instruction_following": n, "readability": n}}, "probes": [{{"q": 1, "coverage": "...", "answer": "..."}}],
"notable_errors": ["..."]}}"""

GROUND = """For each claim below, decide whether the EVIDENCE supports it.
"supported" = the evidence states it (numbers must match); "partial" = some of it is supported, or the claim
over-generalizes; "unsupported" = the evidence does not state it or contradicts it.
<evidence source="{sid}">
{evidence}
</evidence>
Claims:
{claims}
Return ONLY JSON: {{"verdicts": [{{"i": 1, "verdict": "supported|partial|unsupported"}}]}}"""

PROBE_CHECK = """Ground truth for a question, and an answer extracted from a report.
Question: {q}
Ground truth: {gt}
Report answer: {a}
Is the report answer correct? Return ONLY JSON: {{"verdict": "correct|partially_correct|wrong|not_answered"}}"""


def llm_json(prompt: str, max_tokens: int = 4000) -> dict:
    r = _br.converse(modelId=JUDGE_MODEL, messages=[{"role": "user", "content": [{"text": prompt}]}],
                     inferenceConfig={"maxTokens": max_tokens, "temperature": 0})
    txt = r["output"]["message"]["content"][0]["text"]
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0)) if m else {}


def strip_sources(report: str) -> str:
    return report.split("\n## Sources")[0]


def claims_with_citations(report: str) -> list[tuple[str, str]]:
    body = strip_sources(report)
    pairs = []
    for sent in re.split(r"(?<=[。．.!?！？])\s+|\n", body):
        sids = re.findall(r"\[(S\d+)\]", sent)
        clean = re.sub(r"\[S\d+\]", "", sent).strip(" -*|")
        if len(clean) < 15:
            continue
        for sid in dict.fromkeys(sids):
            pairs.append((clean[:600], sid))
    return pairs


def refetch(url: str) -> str:
    import httpx
    import trafilatura
    try:
        r = httpx.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (compatible; DREval-judge)"})
        if "pdf" in r.headers.get("content-type", ""):
            import io
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(r.content)).pages[:40])
        return trafilatura.extract(r.text, include_tables=True, favor_recall=True) or ""
    except Exception:
        return ""


def ground(pairs, sources_by_sid, k: int, seed: str, live: bool) -> dict:
    rnd = random.Random(seed)
    sample = pairs if len(pairs) <= k else rnd.sample(pairs, k)
    by_sid = defaultdict(list)
    for c, sid in sample:
        by_sid[sid].append(c)
    counts = {"seen": defaultdict(int), "page": defaultdict(int)}
    for sid, claims in by_sid.items():
        src = sources_by_sid.get(sid)
        if not src:
            counts["seen"]["invalid_id"] += len(claims)
            continue
        claim_txt = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
        evid = "\n---\n".join(src.get("evidence") or [])[:24000] or "(no text seen)"
        v = llm_json(GROUND.format(sid=sid, evidence=evid, claims=claim_txt))
        for x in v.get("verdicts", []):
            counts["seen"][x.get("verdict", "unsupported")] += 1
        if live:
            page = refetch(src["url"])[:24000]
            if not page:
                counts["page"]["unreachable"] += len(claims)
                continue
            v = llm_json(GROUND.format(sid=sid, evidence=page, claims=claim_txt))
            for x in v.get("verdicts", []):
                counts["page"][x.get("verdict", "unsupported")] += 1

    def rate(c):
        n = sum(v for k2, v in c.items() if k2 in ("supported", "partial", "unsupported", "invalid_id"))
        return round((c.get("supported", 0) + 0.5 * c.get("partial", 0)) / n, 3) if n else None
    return {"n_pairs_total": len(pairs), "n_sampled": len(sample),
            "seen": dict(counts["seen"]), "seen_support_rate": rate(counts["seen"]),
            "page": dict(counts["page"]), "page_support_rate": rate(counts["page"]) if live else None,
            "est_effective_citations": round(len(pairs) * (rate(counts["seen"]) or 0), 1)}


def score_run(run_dir: str, topic_cfg: dict, gt: dict, k: int, live: bool) -> dict:
    summ = json.load(open(os.path.join(run_dir, "summary.json")))
    report = open(os.path.join(run_dir, "report.md"), encoding="utf-8").read() \
        if os.path.exists(os.path.join(run_dir, "report.md")) else ""
    sources = json.load(open(os.path.join(run_dir, "sources.json"))) if os.path.exists(os.path.join(run_dir, "sources.json")) else []
    probes = topic_cfg.get("focus_questions", [])
    res = {"label": summ.get("run_label"), "summary": summ}
    if not report:
        return res
    rub = llm_json(RUBRIC.format(topic=topic_cfg["topic"], report=strip_sources(report)[:120000],
                                 probes="\n".join(f"{i + 1}. {q}" for i, q in enumerate(probes))), 6000)
    res["rubric"] = rub
    # probe correctness against your ground truth (optional)
    tgt = (gt or {}).get(topic_cfg["id"], {})
    checks = []
    for p in rub.get("probes", []):
        i = int(p.get("q", 0)) - 1
        truth = tgt.get(i + 1) or tgt.get(str(i + 1))
        if truth and 0 <= i < len(probes):
            checks.append(llm_json(PROBE_CHECK.format(q=probes[i], gt=truth, a=p.get("answer", ""))).get("verdict"))
    res["probe_correctness"] = checks
    res["grounding"] = ground(claims_with_citations(report), {s["sid"]: s for s in sources}, k,
                              summ.get("run_id", run_dir), live)
    return res


def flatten(r: dict) -> dict:
    s = r["summary"]
    rub = r.get("rubric", {}).get("scores", {})
    probes = r.get("rubric", {}).get("probes", [])
    out = s.get("tool_outcomes", {})
    fetch_total = sum(v for kk, v in out.items() if kk.startswith("fetch_url:") and not kk.endswith(("blocked_scope", "budget")))
    fetch_fail = sum(v for kk, v in out.items() if kk.startswith("fetch_url:") and kk.split(":")[1]
                     not in ("ok", "blocked_scope", "budget"))
    br_total = sum(v for kk, v in out.items() if kk.startswith("browser_") and not kk.endswith("budget"))
    br_block = sum(v for kk, v in out.items() if kk.startswith("browser_") and kk.endswith(("blocked_bot", "error", "empty")))
    cs = s.get("citation_stats", {}) or {}
    g = r.get("grounding", {})
    pc = r.get("probe_correctness", [])
    row = {
        "label": r["label"], "topic": r["label"].split("/")[0], "mode": s["mode"],
        "budget_mode": s.get("budget_mode") or "none", "status": s["status"],
        "elapsed_min": round(s["elapsed_s"] / 60, 1),
        "total_tokens": s["token_usage_total"].get("totalTokens"),
        "est_cost_usd": (s.get("est_cost_usd") or {}).get("total"),
        "limit_hit": 1 if s.get("limits_hit") else 0,  # run cut short by its cost/time cap
        # budget factor: tokens each researcher saw vs its budget, and how much work it did
        "budget_ratio_mean": s.get("budget_ratio_mean"),
        "budget_overruns": s.get("budget_overruns"),
        "researcher_seen_tokens": (round(st.mean(b["counted"] for b in s["researcher_budgets"].values()))
                                   if s.get("researcher_budgets") else None),
        "researcher_tool_calls": (round(st.mean(s["researcher_tool_calls"].values()), 1)
                                  if s.get("researcher_tool_calls") else None),
        "input_tokens": s["token_usage_total"].get("inputTokens"),
        "web_search_queries": s.get("web_search_queries", 0),
        "fetch_calls": fetch_total, "fetch_fail_rate": round(fetch_fail / fetch_total, 3) if fetch_total else None,
        "blocked_scope": out.get("fetch_url:blocked_scope", 0),
        "browser_actions": br_total, "browser_block_rate": round(br_block / br_total, 3) if br_total else None,
        "sources_seen": s.get("sources_seen"), "sources_read_fully": s.get("sources_read_fully"),
        "unique_sources_cited": cs.get("unique_sources_cited"), "cited_read_fully": cs.get("cited_sources_read_fully"),
        "cited_pdfs": cs.get("cited_pdfs"), "cited_domains": cs.get("cited_unique_domains"),
        "invalid_citation_ids": len(cs.get("invalid_citation_ids", []) or []),
        "seen_support_rate": g.get("seen_support_rate"), "page_support_rate": g.get("page_support_rate"),
        "est_effective_citations": g.get("est_effective_citations"),
        "probe_full_rate": round(sum(p.get("coverage") == "full" for p in probes) / len(probes), 3) if probes else None,
        "probe_correct_rate": round(sum(v == "correct" for v in pc) / len(pc), 3) if pc else None,
    }
    for kk, v in rub.items():
        row[f"q_{kk}"] = v
    if rub:
        row["q_mean"] = round(st.mean(rub.values()), 2)
    return row


MODES = ["S1", "S2", "S3", "S4"]
BUDGETS = ["none", "native", "client"]


def _cell(vals: list) -> str:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.3g}"
    return f"{st.mean(vals):.3g} ± {st.stdev(vals):.2g}"


def aggregate(rows: list[dict]) -> str:
    keys = ["researcher_seen_tokens", "researcher_tool_calls", "budget_ratio_mean", "budget_overruns", "q_mean", "q_depth", "q_specificity", "probe_full_rate", "probe_correct_rate", "seen_support_rate",
            "page_support_rate", "est_effective_citations", "unique_sources_cited", "cited_read_fully", "cited_pdfs",
            "blocked_scope", "fetch_fail_rate", "browser_block_rate", "total_tokens", "est_cost_usd", "elapsed_min",
            "limit_hit"]
    grp = defaultdict(list)
    for r in rows:
        grp[(r["topic"], r["mode"], r["budget_mode"])].append(r)
    budgets = [b for b in BUDGETS if any(k[2] == b for k in grp)]
    lines = ["# Results by topic x retrieval mode x budget mode (mean ± sd over repeats)", ""]

    # cross-topic view of the budget factor: rows = retrieval mode, columns = budget mode
    lines += ["## Budget mode effect (all topics pooled)", ""]
    for k in ["q_mean", "probe_full_rate", "seen_support_rate", "researcher_seen_tokens", "researcher_tool_calls",
              "budget_ratio_mean", "est_cost_usd"]:
        lines += [f"**{k}**", "", "| mode | " + " | ".join(budgets) + " |", "|---|" + "---|" * len(budgets)]
        for m in MODES:
            cells = [_cell([r.get(k) for (t, mm, b), rs in grp.items() if mm == m and b == bm for r in rs])
                     for bm in budgets]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")

    for topic in sorted({t for t, _, _ in grp}):
        lines += [f"## {topic}", ""]
        for bm in budgets:
            lines += [f"### budget mode: {bm}", "", "| metric | " + " | ".join(MODES) + " |", "|---|" + "---|" * 4]
            for k in keys:
                lines.append(f"| {k} | " + " | ".join(_cell([r.get(k) for r in grp.get((topic, m, bm), [])])
                                                       for m in MODES) + " |")
            lines.append("")
    hit = [r["label"] for r in rows if r.get("limit_hit")]
    if hit:
        lines += ["> Runs cut short by a cost/time cap (compare with care, or raise the cap and re-run): "
                  + ", ".join(hit), ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=25, help="claim-citation pairs sampled per report")
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(HERE, "topics.yaml")))
    topics = {t["id"]: t for t in cfg["topics"] + cfg.get("optional_topics", [])}
    gt_path = os.path.join(HERE, "ground_truth.yaml")
    gt = yaml.safe_load(open(gt_path)) if os.path.exists(gt_path) else {}
    rows = []
    for sp in sorted(glob.glob(os.path.join(RES, "*", "*", "*", "rep*", "summary.json"))):
        run_dir = os.path.dirname(sp)
        tid, mode, bm, rep = run_dir.split(os.sep)[-4:]
        cache = os.path.join(OUT, f"{tid}__{mode}__{bm}__{rep}.json")
        if os.path.exists(cache):  # delete the cache file to re-score a run
            r = json.load(open(cache))
        elif args.aggregate_only:
            continue
        else:
            print("scoring", run_dir)
            r = score_run(run_dir, topics[tid], gt, args.k, args.refetch)
            json.dump(r, open(cache, "w"), ensure_ascii=False, indent=2)
        r["label"] = f"{tid}/{mode}/{bm}/{rep}"  # directory layout is authoritative
        rows.append(flatten(r))
    if not rows:
        print("no runs found under", RES)
        return
    with open(os.path.join(OUT, "runs.csv"), "w", newline="", encoding="utf-8") as f:
        cols = sorted({k for r in rows for k in r},
                      key=lambda c: (c not in ("label", "topic", "mode", "budget_mode"), c))
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = aggregate(rows)
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
