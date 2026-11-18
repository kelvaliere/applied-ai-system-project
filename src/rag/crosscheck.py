"""
Normalize and cross-check what the sources say about one album.

Genre sources: MusicBrainz (weighted by vote count) and Wikidata. Their overlap is the agreement signal.
Flags are shown to the user, never hidden. Rating cross-check needs a second rating source (not in v1).
"""
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ALIASES = {
    "alt rock": "alternative rock", "alternative/indie": "indie", "post punk": "post-punk",
    "intelligent dance music": "idm", "electronica": "electronic", "hiphop": "hip hop", "hip-hop": "hip hop",
    "rap": "hip hop", "trip-hop": "trip hop", "dreampop": "dream pop",
}


def normalize_tag(tag: str) -> str:
    t = re.sub(r"\s+", " ", tag.lower().strip())
    t = re.sub(r" music$", "", t)
    return ALIASES.get(t, t)


@dataclass
class AlbumRecord:
    id: str
    artist: str
    title: str
    year: Optional[str]
    genres: Dict[str, float]            # tag -> weight (0-2): MusicBrainz share + Wikidata presence
    consensus_tags: List[str]           # tags both genre sources agree on
    rating: Optional[float]             # 0-100 (MusicBrainz community rating)
    rating_votes: int
    summary: str
    sources: List[str]
    confidence: float                   # 0-1
    flags: List[str] = field(default_factory=list)
    origin: str = "seed"               # "seed" (curated list) or "llm" (suggested by the LLM, then verified)
    reception: str = ""                # Wikipedia "Critical reception" text (what critics say), may be empty
    cover: str = ""                    # album cover URL, may be empty


def merge(artist: str, title: str, mb: Optional[dict], wiki: Optional[dict], wd: Optional[dict],
          origin: str = "seed", cover: str = "") -> AlbumRecord:
    mb, wiki, wd = mb or {}, wiki or {}, wd or {}
    mb_genres = {normalize_tag(k): v for k, v in (mb.get("genres") or {}).items()}
    top = max(mb_genres.values(), default=1)
    mb_w = {k: v / top for k, v in mb_genres.items()}
    wd_set = {normalize_tag(g) for g in wd.get("genres", [])}

    genres: Dict[str, float] = {}
    for tag in set(mb_w) | wd_set:
        genres[tag] = round(mb_w.get(tag, 0.0) + (1.0 if tag in wd_set else 0.0), 3)
    consensus = sorted(set(mb_w) & wd_set)

    genre_sources = int(bool(mb_w)) + int(bool(wd_set))
    agreement = len(consensus) / min(len(mb_w), len(wd_set)) if mb_w and wd_set else 0.0
    has_summary = bool(wiki.get("extract"))
    confidence = round(0.4 * genre_sources / 2 + 0.4 * agreement + 0.2 * has_summary, 2)

    flags = []
    if genre_sources < 2:
        flags.append("single_genre_source")
    elif agreement < 0.2:
        flags.append("sources_disagree_on_genre")
    if not genres:
        flags.append("no_genre_data")
    if confidence < 0.5:
        flags.append("low_confidence")

    votes = int(mb.get("votes") or 0)
    rating = round(mb["rating"] * 20, 1) if mb.get("rating") and votes > 0 else None
    return AlbumRecord(
        id=f"{re.sub(r'[^a-z0-9]+', '-', artist.lower()).strip('-')}--{re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')}",
        artist=artist, title=title, year=mb.get("year"), genres=genres, consensus_tags=consensus,
        rating=rating, rating_votes=votes, summary=wiki.get("extract", ""),
        sources=[n for n, d in (("musicbrainz", mb_w), ("wikipedia", has_summary), ("wikidata", wd_set)) if d],
        confidence=confidence, flags=flags, origin=origin,
        reception=wiki.get("reception", ""), cover=cover)


def save(records: List[AlbumRecord], path: Path) -> None:
    path.write_text("\n".join(json.dumps(asdict(r), ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


def load(path: Path) -> List[AlbumRecord]:
    return [AlbumRecord(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
