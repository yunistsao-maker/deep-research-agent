"""Prompts. Structure follows the open_deep_research pattern
(brief -> supervisor delegates to parallel researchers -> per-researcher compression -> final report),
re-implemented for Strands + AgentCore. Wording is our own.

Only TOOL_GUIDANCE differs between modes; it describes the capability factually and stays short,
so the prompt itself does not become the experimental variable.
"""

BRIEF_PROMPT = """Today is {today}. You are planning a deep research project.

Topic from the user:
<topic>
{topic}
</topic>

Audience: {audience}
Report language: {language}
Must-answer questions (the final report must address each one explicitly, even if the answer is "not publicly available"):
{focus_questions}
Extra report instructions: {report_instructions}

Write a research brief for a team of researchers. Include:
1. The core question restated precisely, with scope boundaries (time window, geography, technology generation).
2. 4-8 sub-questions that together cover the topic, ordered by importance. Each must be independently researchable.
3. The kinds of primary sources that would settle each sub-question (standards bodies, vendor docs, filings, papers, government data, etc.).
4. Specific facts/figures worth finding (numbers, dates, versions, names) - the things a domain expert would check.
Do not answer the questions yourself. Output only the brief."""

SUPERVISOR_PROMPT = """Today is {today}. You are the lead researcher coordinating a deep research project.

Research brief:
<brief>
{brief}
</brief>

You have one tool, conduct_research(subtopic), which hands a self-contained subtopic to a researcher who
searches the web and returns cited findings. Researchers cannot see the brief or each other, so each
subtopic description must be fully self-contained (state the entity names, time window, and exactly
what to find).

How to work:
- Plan first: decide which sub-questions to delegate. Delegate independent subtopics in parallel by
  calling conduct_research several times in the same turn (at most {max_parallel} at once).
- You may hire at most {max_researchers} researchers in total. Use at most {max_rounds} rounds of delegation.
- After results come back, check them against the brief and the must-answer questions. Delegate a
  follow-up only for a specific, important gap or a contradiction between sources.
- Stop when the brief is adequately covered or the budget is used. Then reply with a short note
  listing which must-answer questions are answered, partially answered, or unanswered. Do not write the report."""

RESEARCHER_PROMPT = """Today is {today}. You are a meticulous researcher. Research this subtopic:
<subtopic>
{subtopic}
</subtopic>

Tools:
{tool_guidance}

Rules:
- {budget_rule} Plan queries carefully; prefer specific queries
  (entity + attribute + year) over broad ones. Use different phrasings/languages (English and Traditional
  Chinese where the topic is Taiwan-related) when results are thin.
- Every source has an ID like [S12]. Cite the ID right after each fact you report.
- Only state numbers, dates, versions and quotes that appear in evidence you actually saw. If you only
  saw a partial figure or a claim without its context, say so. Never guess.
- Prefer primary sources (standards, official documentation, filings, papers, government data) over
  secondary news or blogs; note when only secondary sources were available.
- Stop when you can answer the subtopic well, or when the budget runs out.

Finish with your findings: a detailed, structured write-up of everything relevant you learned (facts,
figures, dates, disagreements between sources, open gaps), with [Sx] citations on every fact."""

# Budget rule (second experimental factor). native and client get the identical wording; only how the
# remaining budget is shown differs (server-side countdown vs. a line appended to each tool result).
BUDGET_RULE = {
    "none": "You have a budget of {max_tool_calls} tool calls.",
    "token": ("You have a token budget of about {token_budget:,} tokens for this subtopic: your own output and "
              "every tool result count against it, and you will see how much remains as you work. Pace yourself: "
              "get the most important facts first and finish with your findings before the budget runs out."),
}

TOOL_GUIDANCE = {
    "S1": "- web_search: search an Amazon-operated web index. Each result gives a title, URL, date and a short\n"
          "  query-relevant snippet. You cannot open pages; you only see snippets.",
    "S2": "- web_search: search an Amazon-operated web index (title, URL, date, short snippet per result).\n"
          "- fetch_url: read the full text of a page or PDF. Only URLs that appeared in your web_search\n"
          "  results can be opened.",
    "S3": "- web_search: search an Amazon-operated web index (title, URL, date, short snippet per result).\n"
          "- fetch_url: read the full text of any public page or PDF, including links you find inside pages\n"
          "  (for example the PDF behind a landing page).",
    "S4": "- web_search: search an Amazon-operated web index (title, URL, date, short snippet per result).\n"
          "- browser_search: search through a real cloud browser on a public search engine (Bing); you see the\n"
          "  engine's result titles, URLs and snippets. You cannot open pages; you only see snippets.",
    "S5": "- web_search: search an Amazon-operated web index (title, URL, date, short snippet per result).\n"
          "- browser_search / browser_open / browser_click / browser_read: a real cloud browser. Search through a\n"
          "  public search engine, open pages (JavaScript runs), click through, and read the rendered text.\n"
          "  Never try to solve CAPTCHAs or log in.",
}

COMPRESS_PROMPT = """You are cleaning up a researcher's notes for a report writer. Below is the researcher's full
working transcript (tool results and notes) for the subtopic:
<subtopic>
{subtopic}
</subtopic>

<transcript>
{transcript}
</transcript>

Rewrite it as a comprehensive findings document:
- Keep EVERY relevant fact, figure, date, name and caveat. Do not summarize away numbers.
- Keep the [Sx] source ID next to each fact. Only use IDs that appear in the transcript.
- Group by theme; note conflicting figures and which source says what.
- End with "Gaps:" listing what could not be found or verified.
Output only the findings document."""

REPORT_PROMPT = """Today is {today}. Write the final research report.

Research brief:
<brief>
{brief}
</brief>

Findings from the research team (with source IDs):
<findings>
{findings}
</findings>

Available sources (ID, title, URL, date):
<sources>
{sources}
</sources>

Requirements:
- Language: {language}. Audience: {audience}. {report_instructions}
- Start with an executive summary (5-8 bullet points with the most decision-relevant findings).
- Then sections that follow the brief's sub-questions. Use tables where they help compare options or numbers.
- Explicitly answer each must-answer question in a section titled "Must-answer questions", one by one;
  say "not found in public sources" when the findings do not support an answer.
- Cite every factual claim with source IDs in square brackets, e.g. [S3] or [S3][S7]. Use only IDs from
  the source list above. Do not invent sources, numbers or URLs; if findings conflict, show both.
- End with "Limitations" (what the evidence could not establish). Do NOT write a Sources/References
  list; it is appended automatically.
Output only the report in Markdown."""
