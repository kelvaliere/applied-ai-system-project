"""
Offline ingest: seed list -> source clients (cached) -> cross-check -> data/albums.jsonl.

    python -m src.rag.ingest            # uses data/raw cache, fetches only what is missing
    python -m src.rag.ingest --refresh  # re-fetch everything (set MB_CONTACT first)
"""
import argparse
import csv
import logging
from pathlib import Path

from . import crosscheck, logging_setup, sources

log = logging.getLogger("rag.ingest")
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEED_CSV = DATA_DIR / "seed_albums.csv"
ALBUMS_JSONL = DATA_DIR / "albums.jsonl"


def build_record(artist: str, title: str, refresh: bool = False, origin: str = "seed", require_mb: bool = False):
    """
    Fetch (cached) + cross-check one album. Returns (record, matched_on_musicbrainz).
    require_mb: stop right after MusicBrainz when it has no match (used to reject LLM suggestions cheaply).
    """
    mb = sources.musicbrainz_album(artist, title, refresh)
    if require_mb and not mb:
        return crosscheck.merge(artist, title, mb, None, None, origin), False
    cover = (sources.deezer_album_cover(artist, title, refresh) or {}).get("cover", "")
    wiki = sources.wikipedia_album(artist, title, refresh)
    wd = sources.wikidata_genres(wiki["qid"], refresh) if wiki and wiki.get("qid") else None
    return crosscheck.merge(artist, title, mb, wiki, wd, origin, cover), bool(mb)


def build(seed_csv: Path = SEED_CSV, out: Path = ALBUMS_JSONL, refresh: bool = False):
    with seed_csv.open(newline="", encoding="utf-8") as f:
        seeds = [(r["artist"].strip(), r["album"].strip()) for r in csv.DictReader(f) if r.get("artist") and r.get("album")]
    records = []
    for artist, title in seeds:
        rec, _ = build_record(artist, title, refresh)
        records.append(rec)
        log.info("%s - %s | sources=%s confidence=%.2f flags=%s", artist, title, rec.sources, rec.confidence, rec.flags)
    crosscheck.save(records, out)
    log.info("wrote %d albums to %s", len(records), out)
    return records


if __name__ == "__main__":
    logging_setup.setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-fetch")
    build(refresh=ap.parse_args().refresh)
