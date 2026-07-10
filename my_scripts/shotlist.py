"""
Hand-authored B-roll shot list for hydration.txt — v2, phrase queries.

Each scene gets a list of QUERIES tried in order:
  1. a natural-language phrase describing the exact shot wanted
     (Pexels understands these far better than keyword triples)
  2. a simpler backup phrase
  3. a single reliable anchor word (never fails)

Recurring motifs (glass of water, dark bedroom, tired man, kidneys, coffee)
keep the video visually cohesive, like a human-edited health documentary.
"""
import json
import sys

SHOTS = {
    1: ["city at night aerial view", "night city lights", "city"],
    2: ["man drinking glass of water at home", "person drinking water", "water"],
    3: ["taking medicine pill with glass of water", "hand holding pills", "medicine"],
    4: ["pouring glass of water at night kitchen", "glass of water on table", "water"],
    5: ["man waking up in bed at night", "man asleep in bed", "sleeping"],
    6: ["walking to bathroom at night hallway", "bathroom light at night", "bathroom"],
    7: ["alarm clock on bedside table at night", "digital clock night", "clock"],
    8: ["tired senior man sitting on bed", "sad elderly man portrait", "senior"],
    9: ["doctor talking with senior patient", "doctor consultation office", "doctor"],
    10: ["pouring water into drinking glass", "water pouring glass close up", "water"],
    11: ["man drinking glass of water", "person drinking water home", "water"],
    12: ["glass of water in morning sunlight", "fresh water glass window light", "water"],
    13: ["woman drinking water smiling healthy", "young woman drinking water", "water"],
    14: ["swollen ankles feet close up", "massaging swollen leg", "feet"],
    15: ["man thinking serious face close up", "pensive man portrait", "thinking"],
    16: ["bathroom door open at night", "dark bathroom light on", "bathroom"],
    17: ["man sleeping peacefully in bed", "person asleep bed night", "sleeping"],
    18: ["man lying awake in bed staring ceiling", "man insomnia bed", "insomnia"],
    19: ["alarm clock beside sleeping person", "bedside clock lamp night", "clock"],
    20: ["walking into bathroom turning on light", "bathroom at night", "bathroom"],
    21: ["kidney human anatomy 3d animation", "kidney medical illustration", "kidney"],
    22: ["pouring whiskey alcohol into glass evening", "salt shaker food close up", "alcohol"],
    23: ["man rushing to bathroom urgent", "man opening bathroom door", "bathroom"],
    24: ["doctor writing medical chart clipboard", "doctor examining patient", "doctor"],
    25: ["man confused thinking at night", "man pondering dark room", "thinking"],
    26: ["man lying awake in bed eyes open", "person cannot sleep bed night", "awake"],
    27: ["woman with insomnia in bed at night", "person stressed in bed", "insomnia"],
    28: ["man awake in dark bedroom", "man sitting up in bed night", "bedroom"],
    29: ["doctor and patient conversation clinic", "medical consultation talking", "doctor"],
    30: ["man serious thinking portrait", "close up thoughtful face man", "thinking"],
    31: ["pouring fresh water into glass", "water glass filling close up", "water"],
    32: ["clean drinking water glass on table", "clear water glass simple", "water"],
    33: ["sunrise through bedroom window morning", "morning sun light window", "sunrise"],
    34: ["woman drinking water in morning", "person drinking water daylight", "water"],
    35: ["water splashing in glass slow motion", "water splash close up", "water"],
    36: ["morning sunlight in bright room", "sun rays through window", "morning"],
    37: ["drinking coffee in the morning", "coffee cup morning table", "coffee"],
    38: ["pouring coffee into cup morning", "coffee cup breakfast", "coffee"],
    39: ["busy office people working", "person working at desk computer", "office"],
    40: ["family dinner table evening meal", "eating dinner at home", "dinner"],
    41: ["kidney anatomy medical 3d render", "human kidneys illustration", "kidney"],
    42: ["blood cells flowing 3d animation", "blood flow medical animation", "blood"],
    43: ["kidney filtering medical animation", "kidney human body 3d", "kidney"],
    44: ["glass of water on nightstand bedroom", "water glass beside bed", "water"],
    45: ["sunrise over city morning sky", "morning sky sun rising", "sunrise"],
    46: ["pouring water into glass kitchen morning", "filling water glass", "water"],
    47: ["measuring water in glass", "glass of water on kitchen counter", "water"],
    48: ["hand holding glass of drinking water", "glass water in hand", "water"],
    49: ["drinking from large water bottle", "big water bottle gym", "bottle"],
    50: ["water overflowing from full glass", "water spilling glass slow motion", "water"],
    51: ["woman sleeping peacefully at night", "person deep sleep bed", "sleeping"],
    52: ["healthy breakfast with water and fruit", "breakfast table morning food", "breakfast"],
    53: ["happy woman holding glass of water", "smiling person drinking water", "water"],
    54: ["people walking city street daytime", "busy people walking outdoors", "people"],
    55: ["woman sleeping soundly in bed", "peaceful deep sleep night", "sleeping"],
    56: ["man drinking water while working", "person drinking water desk", "water"],
    57: ["pouring water into several glasses", "water bottle pouring glass", "water"],
    58: ["chugging water bottle fast", "man drinking whole water bottle", "drinking"],
    59: ["calm water glass on wooden table", "still water in glass", "water"],
    60: ["lunch meal with glass of water", "healthy lunch table water", "lunch"],
    61: ["diverse group of different people", "crowd of people faces", "people"],
    62: ["doctor health checkup examination", "nurse checking patient health", "doctor"],
    63: ["desert heat sun blazing", "hot desert landscape sun", "desert"],
    64: ["hospital patient with doctor care", "doctor hospital ward patient", "hospital"],
    65: ["clock hands moving time lapse", "analog clock face close up", "clock"],
    66: ["water bottle on table in evening", "evening kitchen water glass", "evening"],
    67: ["woman drinking water at lunch outdoors", "person drinking water restaurant", "water"],
    68: ["runner drinking water after exercise", "athlete drinking water sweat", "exercise"],
    69: ["wall clock evening living room", "clock at dusk home", "clock"],
    70: ["small glass of water on table", "half glass of water", "glass"],
    71: ["bedroom clock at night ten thirty", "bedside table lamp clock night", "clock"],
    72: ["evening dinner time clock", "clock in warm evening light", "evening"],
    73: ["taking small sip of water", "person sipping water glass", "sip"],
    74: ["small sip from glass of water", "woman sipping water slowly", "sip"],
    75: ["taking pill with sip of water night", "medicine pill bedside night", "pills"],
    76: ["water bottle on nightstand at night", "bedside table water bottle lamp", "nightstand"],
    77: ["man drinking water in bed at night", "drinking water middle of night", "drinking"],
    78: ["man waking up tired at night", "person awake night bedroom", "awake"],
    79: ["exhausted man rubbing eyes at night", "tired man face night", "tired"],
    80: ["man cannot sleep tossing in bed", "restless sleep tossing turning", "sleepless"],
    81: ["small glass of water bedside", "tiny sip water night", "glass"],
    82: ["glass of water on wooden table", "water glass simple background", "water"],
    83: ["sipping water slowly close up", "slow sip from water glass", "sip"],
    84: ["man gulping water fast thirsty", "person chugging glass of water", "drinking"],
    85: ["slowly sipping from water glass", "small sips of water", "sip"],
    86: ["person taking small sip water waiting", "thoughtful sip of water", "sip"],
    87: ["man with dry mouth thirsty", "thirsty man licking lips", "thirsty"],
    88: ["confused man shrugging thinking", "man puzzled expression", "confused"],
    89: ["man snoring sleeping mouth open", "person snoring in bed", "snoring"],
    90: ["pouring lots of water into glass", "filling large glass water", "water"],
    91: ["tired man waking up morning headache", "man rubbing face morning tired", "tired"],
    92: ["man breathing heavily while sleeping", "person sleeping mouth open", "breathing"],
    93: ["man with sleep apnea cpap mask", "sleep apnea machine night", "sleep"],
    94: ["chest breathing during sleep close up", "person chest rising sleeping", "breathing"],
    95: ["sleeping man with heart rate monitor", "sleep study patient monitor", "sleep"],
    96: ["frustrated man awake at night bed", "man angry cannot sleep", "frustrated"],
    97: ["water bottle standing on table", "single water bottle plain", "bottle"],
    98: ["sleep clinic study monitoring patient", "medical sleep laboratory", "clinic"],
    99: ["coffee tea and drinks on table", "different beverages cups table", "drinks"],
    100: ["pouring hot coffee into cup", "espresso pouring cup close up", "coffee"],
    101: ["coffee beans and steaming cup", "hot coffee cup steam", "coffee"],
    102: ["drinking coffee in evening", "coffee cup at night table", "coffee"],
    103: ["woman lightly sleeping restless", "light sleep waking easily", "sleeping"],
    104: ["drinking coffee in afternoon office", "afternoon coffee break", "coffee"],
    105: ["coffee cup next to clock", "coffee break time clock", "coffee"],
    106: ["morning coffee at sunrise", "coffee cup early morning light", "coffee"],
    107: ["calendar pages flipping days", "marking calendar days", "calendar"],
    108: ["pouring hot tea into cup", "tea kettle pouring teacup", "tea"],
    109: ["senior woman drinking tea evening", "elderly person tea cup", "tea"],
}

if __name__ == "__main__":
    path = sys.argv[1]
    plan = json.load(open(path))
    missing = [s["scene"] for s in plan if s["scene"] not in SHOTS]
    if missing:
        raise SystemExit(f"No shot authored for scenes: {missing}")
    for s in plan:
        s["keywords"] = SHOTS[s["scene"]]
    json.dump(plan, open(path, "w"), indent=2)
    print(f"Applied {len(SHOTS)} authored phrase-query shots to {path}")
