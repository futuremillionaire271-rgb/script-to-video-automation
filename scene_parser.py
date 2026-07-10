"""
Scene splitting and keyword extraction for script-to-video.

Step 1: Break script into ~4-second scenes (respecting sentence boundaries).
Step 2: Extract 2-3 visual keywords per scene.
"""

import re
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

# Download required NLTK data if not present
for resource in ['punkt_tab', 'punkt', 'averaged_perceptron_tagger', 'stopwords']:
    try:
        nltk.data.find(resource if '/' in resource else f'tokenizers/{resource}')
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass


SPEAKING_PACE = 2.5  # words per second
TARGET_SCENE_DURATION = 4.0  # seconds


class Scene:
    """Represents a single scene with text, timing, and visual keywords."""

    def __init__(self, text: str, start_time: float, end_time: float, keywords: List[str]):
        self.text = text.strip()
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.keywords = keywords

    def __repr__(self) -> str:
        return (
            f"Scene(start={self.start_time:.2f}s, end={self.end_time:.2f}s, "
            f"duration={self.duration:.2f}s, keywords={self.keywords})"
        )


def load_script(filepath: str) -> str:
    """Load script from a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def split_into_scenes(script: str) -> List[Scene]:
    """
    Split script into ~4-second scenes, respecting sentence boundaries.

    Algorithm:
    1. Split into sentences.
    2. Group sentences into "buckets" targeting ~4s each.
    3. If a sentence alone would exceed 4s, give it its own scene.
    4. If multiple short sentences fit within 4s, group them.
    5. Extract keywords from each scene's text.
    """
    sentences = sent_tokenize(script.strip())
    if not sentences:
        return []

    scenes = []
    current_bucket = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        # Estimate time this sentence would take to speak
        sentence_duration = word_count / SPEAKING_PACE

        # If adding this sentence would exceed the target, finalize current bucket
        if (
            current_bucket
            and current_word_count / SPEAKING_PACE + sentence_duration > TARGET_SCENE_DURATION
        ):
            # Finalize the current bucket
            scene_text = ' '.join(current_bucket)
            keywords = extract_keywords(scene_text)
            current_word_count_final = sum(len(s.split()) for s in current_bucket)
            duration = current_word_count_final / SPEAKING_PACE
            scenes.append((scene_text, duration, keywords))

            # Start a new bucket with this sentence
            current_bucket = [sentence]
            current_word_count = word_count
        else:
            # Add sentence to current bucket
            current_bucket.append(sentence)
            current_word_count += word_count

    # Finalize the last bucket
    if current_bucket:
        scene_text = ' '.join(current_bucket)
        keywords = extract_keywords(scene_text)
        current_word_count_final = sum(len(s.split()) for s in current_bucket)
        duration = current_word_count_final / SPEAKING_PACE
        scenes.append((scene_text, duration, keywords))

    # Convert to Scene objects with absolute timing
    scene_objects = []
    current_time = 0.0
    for scene_text, duration, keywords in scenes:
        start_time = current_time
        end_time = current_time + duration
        scene_objects.append(Scene(scene_text, start_time, end_time, keywords))
        current_time = end_time

    return scene_objects


def extract_keywords(text: str) -> List[str]:
    """
    Extract 2-3 visual keywords from scene text.

    Strategy:
    1. Tokenize and POS-tag the text.
    2. Extract nouns (NN, NNS, NNP, NNPS) and adjectives (JJ, JJR, JJS).
    3. Filter out common stop words.
    4. Return top 3 by frequency/importance (or fewer if text is short).
    """
    stop_words_set = set(stopwords.words('english'))

    tokens = word_tokenize(text.lower())
    # Remove punctuation-only tokens
    tokens = [t for t in tokens if re.match(r'\w', t)]

    # POS tagging to identify nouns and adjectives
    pos_tags = pos_tag(tokens)

    # Extract nouns and adjectives, filter stopwords
    candidates = [
        word for word, pos in pos_tags
        if pos in ('NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS')
        and word not in stop_words_set
        and len(word) > 2  # Skip very short words
    ]

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            unique_candidates.append(c)
            seen.add(c)

    # Return top 3 keywords
    return unique_candidates[:3]


def print_scenes(scenes: List[Scene]) -> None:
    """Pretty-print scene breakdown for inspection."""
    if not scenes:
        print("No scenes generated.")
        return

    print(f"\n{'='*80}")
    print(f"Scene Breakdown ({len(scenes)} scenes)")
    print(f"{'='*80}\n")

    total_duration = 0.0
    for i, scene in enumerate(scenes, 1):
        print(f"Scene {i} | {scene.start_time:06.2f}s → {scene.end_time:06.2f}s "
              f"({scene.duration:5.2f}s)")
        print(f"  Text: {scene.text[:70]}..." if len(scene.text) > 70 else f"  Text: {scene.text}")
        print(f"  Keywords: {', '.join(scene.keywords)}")
        print()
        total_duration += scene.duration

    print(f"{'='*80}")
    print(f"Total duration: {total_duration:.2f}s ({total_duration/60:.2f}m)")
    print(f"{'='*80}\n")
