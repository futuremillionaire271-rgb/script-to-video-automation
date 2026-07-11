"""
Script analyzer — turns narration lines into STRONG stock-search queries.

The core insight: if the query is strong, the first search results already
match, and no amount of post-filtering can rescue a weak query. So the
intelligence lives here:

1. CONCEPT -> VISUAL translation: a curated dictionary mapping the ideas a
   narrator says ("caffeine", "sleep apnea", "swollen ankles") to shots a
   stock library actually has ("pouring coffee into cup", "man sleeping
   with cpap mask", "swollen ankles legs close up"). Longest match wins.
2. ABSTRACT-WORD suppression: words like "solution/problem/reason/way"
   describe ideas, not pictures — they never enter a query.
3. SUBJECT + ACTION composition: when the line has a filmable actor and
   verb ("they drink ... in the evening"), compose a natural phrase the
   way a human would search ("person drinking glass of water evening").
4. TOPIC CARRY-FORWARD: an abstract line ("That creates a double problem.")
   inherits the running visual topic — exactly what a human editor cuts to.

Output per scene: [primary phrase, secondary phrase, anchor word] — the
same ladder the search pipeline consumes.
"""

import re
from collections import Counter

# ---------------------------------------------------------------------------
# Knowledge: concepts -> filmable shots. Multi-word keys match first.
# Keys are regexes matched on the lowercased scene text.
# ---------------------------------------------------------------------------
CONCEPT_VISUALS: list[tuple[str, str, str]] = [
    # (pattern, visual phrase, anchor word)
    # -- sleep & night
    (r"sleep apnea|breathing (repeatedly )?stops|pauses in breathing",
     "man sleeping with cpap mask sleep apnea", "sleep"),
    (r"snor(e|es|ing)", "man snoring sleeping mouth open", "snoring"),
    (r"insomnia|struggle to fall asleep|cannot sleep|sleepless|lying awake|lie awake|wakes? (them|you|me) up|waking up|woke you",
     "man lying awake in bed at night", "insomnia"),
    (r"deep sleep|sleep(ing)? (soundly|peacefully)|asleep",
     "person sleeping peacefully in bed", "sleeping"),
    (r"bedtime|go(ing)? to bed|before bed", "person getting into bed at night", "bedtime"),
    (r"nightstand|bedside", "nightstand water bottle lamp night", "nightstand"),
    (r"alarm clock|two in the morning|before morning|middle of the night",
     "alarm clock on bedside table at night", "clock"),
    (r"dream|nightmare", "person sleeping restless night", "sleeping"),
    # -- drinking & liquids
    (r"glass(es)? of water|large glass|small glass|sixteen.ounce|glass with",
     "pouring glass of water close up", "water"),
    (r"sip(s|ping)?\b", "person taking small sip of water", "sip"),
    (r"gulp|chug|rapid(ly)? (drink|swallow)|large swallows|entire bottle",
     "man chugging water bottle fast thirsty", "drinking"),
    (r"drink(s|ing)? water|stay(ing)? hydrated|hydration|consume water|fluid intake|water enters",
     "person drinking glass of water at home", "water"),
    (r"water bottle|bottle\b", "water bottle on table", "bottle"),
    (r"\btea\b", "pouring hot tea into cup steam", "tea"),
    (r"coffee|caffeine|caffeinated", "pouring hot coffee into cup", "coffee"),
    (r"alcohol|whiskey|wine|beer", "pouring whiskey alcohol into glass", "alcohol"),
    (r"cola|soda|energy drink", "glass of cola soda with ice", "soda"),
    (r"chocolate", "hot chocolate drink cup", "chocolate"),
    (r"salt\b|salty", "salt shaker pouring close up", "salt"),
    (r"\bwater\b", "fresh drinking water glass", "water"),
    (r"\bdrink(s|ing)?\b", "different drinks in glasses on table", "drinks"),
    # -- bathroom
    (r"bathroom|urinat|toilet|bladder empt|visit the bathroom|pee\b",
     "walking to bathroom at night hallway light", "bathroom"),
    (r"bladder", "human bladder anatomy medical illustration", "medical"),
    # -- body & medicine
    (r"kidney", "kidney human anatomy 3d medical animation", "kidney"),
    (r"prostate", "doctor consultation older man clinic", "doctor"),
    (r"hormone", "medical laboratory research scientist", "laboratory"),
    (r"diabetes|blood sugar", "glucose meter blood sugar test diabetes", "diabetes"),
    (r"blood(stream)?\b", "blood cells flowing 3d medical animation", "blood"),
    (r"swollen|swelling|ankle", "swollen ankles legs close up", "legs"),
    (r"heart (failure|disease)|advanced kidney", "hospital patient with doctor care", "hospital"),
    (r"medicat|medication|pills?\b|supplement", "taking pill medication with glass of water", "medicine"),
    (r"thirst(y|ier)?", "thirsty man drinking water", "thirsty"),
    (r"doctor|evaluation|medical|clinic", "doctor talking with patient consultation", "doctor"),
    (r"infection|constipation|irritation", "man stomach pain discomfort", "pain"),
    (r"headache", "man morning headache rubbing temples", "headache"),
    (r"dry mouth|mouth (is )?drying|dry bedroom", "thirsty man dry lips drinking", "thirsty"),
    (r"mouth breathing|nasal congestion|gasp", "man sleeping mouth open breathing", "breathing"),
    (r"pain\b|anxiety|stress", "stressed man rubbing forehead", "stressed"),
    (r"exhaust|tired|sleepiness|fatigue", "tired man rubbing eyes exhausted", "tired"),
    # -- time of day
    (r"first hour of the morning|after waking|wake up in the morning|start the morning",
     "man waking up morning stretching bed", "morning"),
    (r"morning", "morning sunlight through bedroom window", "morning"),
    (r"afternoon", "afternoon office people working", "afternoon"),
    (r"evening|dinner|after dinner", "family dinner table evening warm light", "evening"),
    (r"lunch(time)?", "healthy lunch meal with water", "lunch"),
    (r"breakfast", "breakfast table morning coffee", "breakfast"),
    (r"overnight|at night|nighttime|per night|during the night",
     "dark bedroom at night moonlight", "night"),
    (r"three to four hours|ten thirty|six thirty|seven thirty|clock",
     "clock on wall evening time", "clock"),
    (r"several days|one to two weeks|calendar", "calendar pages flipping days", "calendar"),
    # -- people & places
    (r"millions of people|most people|many people|everyone",
     "crowd of people walking city street", "people"),
    (r"older adults|senior|elderly|with age|aging",
     "senior man portrait thoughtful", "senior"),
    (r"arizona|desert|outdoors in", "desert sun heat working outdoors", "desert"),
    (r"michigan|air.conditioned", "office air conditioned room window", "office"),
    (r"television|watching tv", "person watching television evening couch", "television"),
    (r"exercise|work(ing)? outdoors|sweat", "runner drinking water after exercise", "exercise"),
    (r"busy|forget to drink|remain busy", "busy person working at desk office", "busy"),
    (r"body size|weather|physical activity", "diverse people walking outdoors", "people"),
    # -- abstractions with a standard visual metaphor
    (r"curve\b|rise .* fall|reversed", "line graph curve animation", "graph"),
    (r"cycle\b|self.reinforcing|loop", "man awake drinking water night bed", "night"),
    (r"question\b|\?\"?$", "man thinking question portrait", "thinking"),
    (r"distinction|two very different|completely different",
     "two glasses of water comparison table", "comparison"),
    (r"experiment|observe what happens|judge", "person writing notes journal", "notes"),
    (r"timing|right times", "hourglass sand time close up", "hourglass"),
]

# When a concept repeats across many scenes, rotate its visual so the
# video doesn't show the same kind of shot six times.
VISUAL_ALTERNATES: dict[str, list[str]] = {
    "different drinks in glasses on table": [
        "different drinks in glasses on table",
        "man holding drink glass evening",
        "refreshing drink glass close up",
        "person pouring drink at home",
    ],
    "glass of water on wooden table": [
        "glass of water on wooden table",
        "clear water glass daylight",
        "still water in drinking glass",
    ],
    "hand holding glass of drinking water": [
        "hand holding glass of drinking water",
        "woman holding water glass smiling",
        "close up hand picking up water glass",
    ],
    "man sleeping with cpap mask sleep apnea": [
        "man sleeping with cpap mask sleep apnea",
        "sleep study patient monitoring clinic",
        "man sleeping restless breathing night",
    ],
    "man chugging water bottle fast thirsty": [
        "man chugging water bottle fast thirsty",
        "athlete drinking whole bottle of water",
        "thirsty man gulping water",
    ],
    "pouring hot tea into cup steam": [
        "pouring hot tea into cup steam",
        "tea cup with steam close up",
        "senior woman drinking tea",
    ],
    "kidney human anatomy 3d medical animation": [
        "kidney human anatomy 3d medical animation",
        "doctor showing anatomy on tablet to patient",
        "human body medical scan 3d animation",
    ],
    "human bladder anatomy medical illustration": [
        "human bladder anatomy medical illustration",
        "doctor explaining anatomy chart patient",
        "medical 3d human body animation",
    ],
    "man lying awake in bed at night": [
        "man lying awake in bed at night",
        "woman awake in bed staring at ceiling night",
        "man sitting on edge of bed at night tired",
    ],
    "walking to bathroom at night hallway light": [
        "walking to bathroom at night hallway light",
        "bathroom door light on at night",
        "man walking down dark hallway at night",
    ],
    "person drinking glass of water at home": [
        "person drinking glass of water at home",
        "woman drinking water in kitchen",
        "man drinking glass of water close up",
    ],
    "pouring glass of water close up": [
        "pouring glass of water close up",
        "filling drinking glass with fresh water",
        "water pouring into glass slow motion",
    ],
    "fresh drinking water glass": [
        "fresh drinking water glass",
        "glass of water on wooden table",
        "hand holding glass of drinking water",
    ],
    "pouring hot coffee into cup": [
        "pouring hot coffee into cup",
        "coffee cup steam close up morning",
        "espresso machine pouring coffee",
    ],
    "dark bedroom at night moonlight": [
        "dark bedroom at night moonlight",
        "bedroom at night dim lamp",
        "moonlight through bedroom window",
    ],
    "person getting into bed at night": [
        "person getting into bed at night",
        "man turning off bedside lamp night",
        "woman getting under blanket bed evening",
    ],
    "crowd of people walking city street": [
        "crowd of people walking city street",
        "busy pedestrian crossing timelapse",
        "people walking on sidewalk city",
    ],
}


# Words that describe ideas, not pictures — banned from queries
ABSTRACT_WORDS = {
    "solution", "problem", "reason", "reasons", "way", "ways", "thing",
    "things", "amount", "amounts", "strategy", "principle", "distinction",
    "possibility", "pattern", "rule", "rules", "goal", "advantage",
    "competition", "needs", "point", "terms", "cause", "attention",
    "question", "questions", "answer", "case", "cases", "change", "changes",
    "part", "parts", "hour", "hours", "time", "times", "size", "portion",
    "number", "ounces", "moment", "example", "person", "people",
}

_WORD = re.compile(r"[a-z']+")


def _match_concepts(text: str, limit: int = 2) -> list[tuple[str, str]]:
    """All concept visuals whose pattern appears in the text, in dictionary
    priority order (most specific patterns are listed first)."""
    found = []
    for pattern, visual, anchor in CONCEPT_VISUALS:
        if re.search(pattern, text):
            found.append((visual, anchor))
            if len(found) >= limit:
                break
    return found


def script_topic(full_script: str) -> tuple[str, str]:
    """The script's dominant visual (global fallback for abstract lines)."""
    counts = Counter()
    visuals = {}
    for pattern, visual, anchor in CONCEPT_VISUALS:
        hits = len(re.findall(pattern, full_script.lower()))
        if hits:
            counts[(visual, anchor)] = hits
            visuals[(visual, anchor)] = visual
    if not counts:
        return ("cinematic b-roll landscape", "nature")
    (visual, anchor), _ = counts.most_common(1)[0]
    return (visual, anchor)


def generate_queries(scenes, full_script: str) -> None:
    """
    Assign every scene a strong query ladder [primary, secondary, anchor].
    Mutates scene.keywords in place. Abstract scenes inherit the running
    topic (carry-forward), like a human editor's B-roll continuity.
    """
    topic_visual, topic_anchor = script_topic(full_script)
    running = (topic_visual, topic_anchor)
    usage = Counter()

    for scene in scenes:
        text = scene.text.lower()
        concepts = _match_concepts(text)

        if concepts:
            primary, anchor = concepts[0]
            alts = VISUAL_ALTERNATES.get(primary)
            if alts:
                primary = alts[usage[concepts[0][0]] % len(alts)]
            usage[concepts[0][0]] += 1
            secondary = concepts[1][0] if len(concepts) > 1 else running[0]
            running = (primary, anchor)
        else:
            # No filmable concept: carry the current topic forward
            primary, anchor = running
            secondary = topic_visual

        scene.keywords = list(dict.fromkeys([primary, secondary, anchor]))


def queries_report(scenes) -> str:
    lines = []
    for i, s in enumerate(scenes, 1):
        lines.append(f"{i:3d} | {s.text[:58]:58s} -> {s.keywords[0]}")
    return "\n".join(lines)
