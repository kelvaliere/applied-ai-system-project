"""
Short album descriptions. An LLM (Gemini if GEMINI_API_KEY is set, else Claude if ANTHROPIC_API_KEY is set) writes them
from retrieved records only; a grounding check validates the output; anything missing or invalid falls back to a
deterministic template. Works with no API key. Keys come from the environment or a gitignored .env file.
"""
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
import requests

log = logging.getLogger("rag.generate")
CLAUDE_MODEL = "claude-sonnet-5-5"
GEMINI_MODEL_DEFAULT = "gemini-3.5-flash-lite"   # fast + cheap; override with GEMINI_MODEL (2.5-flash is closed to new users)
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load_env(path: Optional[Path] = None) -> None:
    """Minimal .env reader (KEY=VALUE lines). Real environment variables win; no extra dependency."""
    path = path or ENV_FILE
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def provider() -> Optional[str]:
    load_env()
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    return None
SYSTEM = (
    "You write 1-2 sentence descriptions of albums for a music recommender. Describe the album itself: its sound and "
    "atmosphere, and how critics or listeners received it. Base mood and reception wording on the 'reception' text "
    "when it is present; otherwise use 'summary' and 'genres' and stay factual. Use ONLY facts present in the JSON "
    "you are given. Do not add names, years, scores or claims that are not in it. Do not mention the listener or why "
    "the album was recommended. Reply with a single JSON object mapping each album id to its description, and "
    "nothing else."
)
MAX_BLURB_CHARS = 400


def first_sentence(text: str) -> str:
    m = re.match(r"(.+?[.!?])(\s|$)", text.strip())
    return m.group(1) if m else text.strip()[:200]


def template_blurb(item: dict) -> str:
    genres = ", ".join(item["genres"][:3]) or "unclassified"
    text = item.get("reception") or item.get("summary")
    return first_sentence(text) if text else f"{item['title']} is a {genres} album by {item['artist']}."


def _numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def validate(blurbs: dict, items: List[dict]) -> Dict[str, str]:
    """Keep only blurbs for known ids that are non-empty, short, and contain no number absent from that item's data."""
    by_id = {i["id"]: i for i in items}
    ok = {}
    for album_id, text in blurbs.items():
        item = by_id.get(album_id)
        if item is None or not isinstance(text, str) or not text.strip() or len(text) > MAX_BLURB_CHARS:
            log.warning("grounding: dropped blurb for %r (unknown id, empty or too long)", album_id)
            continue
        if not _numbers(text) <= _numbers(json.dumps(item)):
            log.warning("grounding: dropped blurb for %r (number not in source data)", album_id)
            continue
        ok[album_id] = text.strip()
    return ok


LLM_ERRORS = (anthropic.APIError, requests.RequestException, ValueError, KeyError, IndexError,
              StopIteration, json.JSONDecodeError)


def _parse_json(text: str):
    return json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip())


def _ask_claude(system: str, payload):
    client = anthropic.Anthropic(timeout=30.0, max_retries=2)
    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=4000, system=system,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
    if resp.stop_reason != "end_turn":
        raise ValueError(f"unexpected stop_reason {resp.stop_reason}")
    return _parse_json(next(b.text for b in resp.content if b.type == "text"))


def _ask_gemini(system: str, payload):
    """Gemini REST call. The key goes in a header (never in the URL), so it cannot leak into logs via exceptions."""
    model = os.environ.get("GEMINI_MODEL", GEMINI_MODEL_DEFAULT)
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
    }
    for attempt in (1, 2, 3):
        resp = requests.post(GEMINI_URL.format(model=model), json=body, timeout=30,
                             headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        if resp.status_code in (429, 503) and attempt < 3:      # rate limit / "high demand": brief retry
            time.sleep(2 * attempt)
            continue
        break
    resp.raise_for_status()
    candidate = resp.json()["candidates"][0]
    if candidate.get("finishReason") not in (None, "STOP"):
        raise ValueError(f"unexpected finishReason {candidate.get('finishReason')}")
    return _parse_json("".join(p.get("text", "") for p in candidate["content"]["parts"]))


def _ask_llm(system: str, payload, which: str):
    return _ask_gemini(system, payload) if which == "gemini" else _ask_claude(system, payload)


def ask_json(system: str, payload):
    """Single JSON-in/JSON-out LLM call via the configured provider. Raises ValueError when no key is set."""
    which = provider()
    if which is None:
        raise ValueError("no LLM key configured")
    return _ask_llm(system, payload, which)


def describe(items: List[dict]) -> Tuple[Dict[str, str], bool]:
    """items: dicts with id, artist, title, year, genres, rating, summary, reception. Returns ({id: blurb}, llm_used)."""
    blurbs: Dict[str, str] = {}
    llm_used = False
    if provider() and items:
        try:
            blurbs = validate(ask_json(SYSTEM, items), items)
            llm_used = bool(blurbs)
        except LLM_ERRORS as exc:
            log.warning("LLM description failed via %s (%s: %s); using templates", provider(), type(exc).__name__, exc)
    for item in items:
        if item["id"] not in blurbs:
            blurbs[item["id"]] = template_blurb(item)
    return blurbs, llm_used
