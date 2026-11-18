"""
LLM-assisted discovery: the LLM only SUGGESTS candidate albums; our pipeline decides what is real.

    suggest (LLM) -> drop duplicates -> verify each on MusicBrainz + cross-check -> append to data/discovered.jsonl

A suggestion that MusicBrainz cannot match, or that has no genre data, is dropped (guards against hallucinated albums).
The LLM's own explanations are never shown; ranking and the "why" line stay deterministic. Verified albums are
persisted, so the second search for a similar taste costs nothing.
"""
import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Tuple

from . import crosscheck, generate
from .crosscheck import AlbumRecord
from .ingest import build_record

log = logging.getLogger("rag.discover")
DISCOVERED_JSONL = Path(__file__).resolve().parents[2] / "data" / "discovered.jsonl"
N_SUGGEST = 12          # ask for more than we need: some will not verify
MAX_NEW = 6             # verified albums added per search (each costs several seconds of rate-limited API calls)
MAX_ATTEMPTS = 9        # verification attempts per search
MAX_SECONDS = 40        # wall-clock budget for verification; stop trying new candidates after this
MAX_FIELD = 120

SYSTEM = (
    "You are a music expert helping a recommender find albums. Given a listener's favorite songs, suggest "
    f"{N_SUGGEST} REAL albums, released commercially, that someone with this taste would enjoy. Cover the style of "
    "each song, not only the first. Do not suggest albums by the listed artists. Only name albums you are sure exist. "
    'Reply with a JSON list of objects: {"artist": "...", "album": "..."} and nothing else.'
)


def _key(artist: str, title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", f"{artist} {title}".lower()).strip()


def suggest(songs: List[str]) -> List[Tuple[str, str]]:
    """Ask the LLM for candidates. Returns cleaned (artist, album) pairs; raises generate.LLM_ERRORS on failure."""
    raw = generate.ask_json(SYSTEM, songs)
    if not isinstance(raw, list):
        raise ValueError("LLM did not return a list")
    pairs = []
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("artist"), str) and isinstance(item.get("album"), str):
            artist, album = item["artist"].strip(), item["album"].strip()
            if artist and album and len(artist) <= MAX_FIELD and len(album) <= MAX_FIELD:
                pairs.append((artist, album))
    return pairs


def load_discovered(path: Path = DISCOVERED_JSONL) -> List[AlbumRecord]:
    return crosscheck.load(path) if path.exists() else []


def discover(songs: List[str], known: List[AlbumRecord], path: Path = DISCOVERED_JSONL,
             suggest_fn: Callable[[List[str]], List[Tuple[str, str]]] = suggest,
             build_fn=build_record) -> Tuple[List[AlbumRecord], str]:
    """Returns (newly verified records, short status note for the UI). Never raises on LLM/network failure."""
    try:
        candidates = suggest_fn(songs)
    except generate.LLM_ERRORS as exc:
        log.warning("discovery skipped: %s: %s", type(exc).__name__, exc)
        return [], "Discovery unavailable right now; using the existing catalog."

    seen = {_key(r.artist, r.title) for r in known}
    started = time.monotonic()
    new: List[AlbumRecord] = []
    attempts = dropped = 0
    for artist, title in candidates:
        if len(new) >= MAX_NEW or attempts >= MAX_ATTEMPTS or time.monotonic() - started >= MAX_SECONDS:
            break
        if _key(artist, title) in seen:
            continue
        seen.add(_key(artist, title))
        attempts += 1
        rec, matched = build_fn(artist, title, origin="llm", require_mb=True)
        if not matched or not rec.genres:
            dropped += 1
            log.info("discovery dropped %s - %s (matched=%s genres=%d)", artist, title, matched, len(rec.genres))
            continue
        new.append(rec)

    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in new)
    log.info("discovery: suggested=%d tried=%d verified=%d dropped=%d", len(candidates), attempts, len(new), dropped)
    return new, f"Added {len(new)} new album(s) suggested by AI and verified on MusicBrainz ({dropped} suggestion(s) rejected)."
