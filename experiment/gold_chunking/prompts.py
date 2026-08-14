"""Gold chunking 프롬프트 — 공통 시스템 프롬프트 + 데이터셋별 few-shot."""

# 캐시 키에 포함됨. 프롬프트를 바꾸면 반드시 올려서 캐시를 무효화할 것.
PROMPT_VERSION = "v2"

SYSTEM_PROMPT_BASE = """You are an expert at episodic memory segmentation for conversational data.

You are given the turns of a SINGLE conversation session (all messages occur on
the same day). Your task is to split the session into topic-coherent episodes —
segments that are meaningful and independently memorable. Your core principle is
"default to merging, split cautiously".

### When to split

Start a new segment only when a CLEAR signal appears:
- Substantive topic change: the conversation shifts from one concrete topic to a
  completely unrelated one (e.g., a health concern → weekend travel plans).
- Task/thread completion + new topic: a closing turn ("sounds good, thanks!")
  belongs to its current episode; split only when the NEXT turn opens a genuinely
  unrelated topic.

### Do NOT split for
- Greetings or farewells ("hi", "bye", "thanks") — keep them with the episode
  they serve.
- Transition phrases ("by the way", "oh also", "speaking of") — these usually
  CONTINUE the current episode unless they introduce a major, unrelated topic.
- Follow-up questions, clarifications, or brief reactions on the same topic.
- A shift in aspect within the same topic (e.g., discussing a trip's schedule →
  the same trip's budget stays ONE episode).

### Decision principles
- Merge by default: when in doubt, do not split.
- Content over form: greetings and farewells belong to the episode they serve,
  never to their own segment.
- Process continuity: consecutive turns working toward the same goal (e.g.,
  describe a problem → discuss a fix → confirm the fix) form one episode.
- A segment should make sense read in isolation: if a segment's topic cannot be
  named without referring to the previous segment, it should not be separate.

### Output format

Return a JSON object:
{
  "reasoning": "<one or two sentences explaining every boundary decision>",
  "segments": [
    {"start": <first turn number>, "end": <last turn number>,
     "topic": "<3-7 word label naming what this segment is about>"}
  ]
}

Hard rules for segments:
- Turn numbers are 1-based and refer to the numbered list in the input.
- Segments must tile the session exactly: the first segment starts at 1, the
  last segment ends at N (the final turn), each segment's start is the previous
  segment's end + 1. No gaps, no overlaps, no reordering.
- start <= end for every segment. A single segment {"start": 1, "end": N} means
  the whole session is one episode.
- The topic label must describe the segment's content, not its position
  ("opening chat" is a bad label; "planning a camping trip" is a good label)."""

SYSTEM_PROMPT_BASE_V2 = """You are an expert at episodic memory segmentation for conversational data.

You are given the turns of a SINGLE conversation session (all messages occur on
the same day). Your task is to split the session into topic-coherent episodes —
segments that are meaningful and independently memorable.

### When to split

Start a new segment only when a CLEAR signal appears:
- Substantive topic change: the conversation shifts from one concrete topic to a
  completely unrelated one (e.g., a health concern → weekend travel plans).
- Task/thread completion + new topic: a closing turn ("sounds good, thanks!")
  belongs to its current episode; split only when the NEXT turn opens a genuinely
  unrelated topic.

### Do NOT split for
- Greetings or farewells ("hi", "bye", "thanks") — keep them with the episode
  they serve.
- Transition phrases ("by the way", "oh also", "speaking of") — these usually
  CONTINUE the current episode unless they introduce a major, unrelated topic.
- Follow-up questions, clarifications, or brief reactions on the same topic.
- A shift in aspect within the same topic (e.g., discussing a trip's schedule →
  the same trip's budget stays ONE episode).

### Decision principles
- Content over form: greetings and farewells belong to the episode they serve,
  never to their own segment.
- Process continuity: consecutive turns working toward the same goal (e.g.,
  describe a problem → discuss a fix → confirm the fix) form one episode.
- A segment should make sense read in isolation: if a segment's topic cannot be
  named without referring to the previous segment, it should not be separate.

### Output format

Return a JSON object:
{
  "reasoning": "<one or two sentences explaining every boundary decision>",
  "segments": [
    {"start": <first turn number>, "end": <last turn number>,
     "topic": "<3-7 word label naming what this segment is about>"}
  ]
}

Hard rules for segments:
- Turn numbers are 1-based and refer to the numbered list in the input.
- Segments must tile the session exactly: the first segment starts at 1, the
  last segment ends at N (the final turn), each segment's start is the previous
  segment's end + 1. No gaps, no overlaps, no reordering.
- start <= end for every segment. A single segment {"start": 1, "end": N} means
  the whole session is one episode.
- The topic label must describe the segment's content, not its position
  ("opening chat" is a bad label; "planning a camping trip" is a good label)."""

FEWSHOT_LOCOMO = """### Examples

**Example 1 — two episodes:**
[1] Alice: Hey! How was your doctor's appointment yesterday?
[2] Bob: It went okay. They want me to cut down on caffeine though.
[3] Alice: Ouch, that's rough for you. Switching to decaf?
[4] Bob: Trying to. It's been a headache, literally.
[5] Bob: Oh, did I tell you I finally booked the cabin for the ski trip?
[6] Alice: No way! Which weekend did you get?
[7] Bob: Second weekend of January. You're still coming, right?
[8] Alice: Absolutely, wouldn't miss it.
Output:
{"reasoning": "Turns 1-4 discuss Bob's doctor visit and caffeine advice; turn 5 opens an unrelated ski trip topic that continues to the end.", "segments": [{"start": 1, "end": 4, "topic": "Bob's doctor visit and caffeine reduction"}, {"start": 5, "end": 8, "topic": "booking the January ski trip cabin"}]}

**Example 2 — one episode (no split):**
[1] Alice: What did you end up painting for the art fair?
[2] Bob: A landscape of the lake near my parents' house.
[3] Alice: That sounds beautiful. Oils or acrylics?
[4] Bob: Acrylics. By the way, I struggled so much with the water reflections.
[5] Alice: Reflections are the hardest! Did you use glazing?
[6] Bob: Yes, three thin layers. Turned out better than I hoped.
Output:
{"reasoning": "All turns discuss Bob's art fair painting; the 'by the way' in turn 4 stays on the same painting, so no boundary.", "segments": [{"start": 1, "end": 6, "topic": "Bob's lake painting for the art fair"}]}"""

FEWSHOT_LONGMEMEVAL = """### Examples

**Example 1 — two episodes:**
[1] user: Can you help me write a cover letter for a data analyst position at Meridian Health?
[2] assistant: Of course! Could you tell me about your relevant experience?
[3] user: I spent three years doing SQL reporting and built dashboards in Tableau.
[4] assistant: Great. Here's a draft: "Dear Hiring Manager, ..." (draft follows)
[5] user: That's great, thanks. Now, what's a good beginner routine for strength training?
[6] assistant: A solid beginner routine is full-body workouts three times a week: squats, bench press, rows...
[7] user: How much weight should I start with?
[8] assistant: Start with just the bar or light dumbbells to learn form, then add weight gradually.
Output:
{"reasoning": "Turns 1-4 complete the cover letter task; turn 5 thanks and then opens an unrelated strength training topic.", "segments": [{"start": 1, "end": 4, "topic": "writing a data analyst cover letter"}, {"start": 5, "end": 8, "topic": "beginner strength training routine advice"}]}

**Example 2 — one episode (no split):**
[1] user: My sourdough starter isn't rising. It's been five days.
[2] assistant: A few questions: how often do you feed it, and what's your room temperature?
[3] user: Once a day, and my kitchen is around 65°F.
[4] assistant: 65°F is on the cold side. Try feeding twice daily and keeping it somewhere warmer, like inside the oven with the light on.
[5] user: Okay, also should I switch to whole wheat flour?
[6] assistant: Whole wheat can help — it has more wild yeast and nutrients. Even 50/50 with your current flour will speed things up.
Output:
{"reasoning": "All turns troubleshoot the same sourdough starter problem; turn 5's 'also' is a follow-up within the same task.", "segments": [{"start": 1, "end": 6, "topic": "troubleshooting a sourdough starter not rising"}]}"""

USER_PROMPT_TEMPLATE = """Here are the turns of one conversation session (date: {session_date}).
Split it into topic-coherent episodes following the rules above.

{turns_block}

The session has {n} turns. Return the JSON object only."""

# 타일링 위반 시 1회 정정 요청에 쓰는 메시지
CORRECTION_TEMPLATE = """Your segments do not tile turns 1..{n} exactly (first segment must start at 1, last must end at {n}, each start = previous end + 1, no gaps or overlaps). Return the corrected JSON object only."""

FEWSHOT = {
    "locomo": FEWSHOT_LOCOMO,
    "longmemeval": FEWSHOT_LONGMEMEVAL,
}

# OpenAI Structured Outputs용 strict 스키마
SEGMENTS_SCHEMA = {
    "name": "session_segments",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                        "topic": {"type": "string"},
                    },
                    "required": ["start", "end", "topic"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["reasoning", "segments"],
        "additionalProperties": False,
    },
}


# ── 2차 merge 검토 (타깃 패스) ────────────────────────────────────────
MERGE_SYSTEM = """You are verifying conversation segmentation. You are given two ADJACENT segments from the same session, in order. Decide whether they actually form ONE coherent episode about the same topic or goal — meaning the boundary between them should be removed.

Set merge=true if segment B continues A's topic, responds to or acknowledges A, or works toward the same goal — including the common case where A is just an opening greeting or a brief topic introduction and B is the reply that engages with it.
Set merge=false only if B opens a genuinely different, self-standing topic unrelated to A.

Guiding rule: a lone opening turn (greeting + brief intro) should almost never stand as its own episode; if B addresses it at all, merge."""

MERGE_USER_TEMPLATE = """Session date: {date}

Segment A (topic: "{topic_a}"):
{turns_a}

Segment B (topic: "{topic_b}"):
{turns_b}

Should the boundary between A and B be removed (they are one episode)? Return JSON only."""

MERGE_SCHEMA = {
    "name": "merge_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "merge": {"type": "boolean"},
        },
        "required": ["reasoning", "merge"],
        "additionalProperties": False,
    },
}


def _render_turns(turns: list) -> str:
    return "\n".join(f"{t['speaker']}: {t['content'] or ''}" for t in turns)


def build_merge_messages(date: str, seg_a: dict, seg_b: dict) -> list:
    user = MERGE_USER_TEMPLATE.format(
        date=date or "unknown",
        topic_a=seg_a.get("topic", ""),
        topic_b=seg_b.get("topic", ""),
        turns_a=_render_turns(seg_a["turns"]),
        turns_b=_render_turns(seg_b["turns"]),
    )
    return [
        {"role": "system", "content": MERGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_messages(dataset: str, session_date: str, turns: list) -> list:
    """turns: [{"speaker": ..., "content": ...}] → system+user 메시지."""
    turns_block = "\n".join(
        f"[{i + 1}] {t['speaker']}: {t['content'] or ''}" for i, t in enumerate(turns)
    )
    system = SYSTEM_PROMPT_BASE_V2 + "\n\n" + FEWSHOT[dataset]
    user = USER_PROMPT_TEMPLATE.format(
        session_date=session_date or "unknown",
        turns_block=turns_block,
        n=len(turns),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
