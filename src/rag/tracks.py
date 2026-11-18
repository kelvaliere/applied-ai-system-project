"""Track search with 30-second previews (Deezer public API, no key). Only used to help the user pick songs."""
import logging
from dataclasses import dataclass
from typing import List

from . import sources

log = logging.getLogger("rag.tracks")
DEEZER_SEARCH = "https://api.deezer.com/search"
MAX_QUERY = 120


@dataclass(frozen=True)
class Track:
    title: str
    artist: str
    album: str
    cover: str      # small album cover URL ("" if none)
    preview: str    # 30-second mp3 URL ("" if none)

    @property
    def label(self) -> str:
        return f"{self.artist} - {self.title}"


def _https(url) -> str:
    return url if isinstance(url, str) and url.startswith("https://") else ""


def search_tracks(query: str, limit: int = 6) -> List[Track]:
    """Best-effort: returns [] on bad input, network failure or an API error (the UI then falls back to typing)."""
    q = query.strip()[:MAX_QUERY]
    if len(q) < 2:
        return []
    data = sources._get_json(DEEZER_SEARCH, {"q": q, "limit": max(1, min(limit, 10))})
    if not data or not isinstance(data.get("data"), list):
        log.info("track search returned nothing for %r", q)
        return []
    tracks = []
    for item in data["data"]:
        try:
            tracks.append(Track(title=item["title"], artist=item["artist"]["name"], album=item["album"]["title"],
                                cover=_https(item["album"].get("cover_small")), preview=_https(item.get("preview"))))
        except (KeyError, TypeError, AttributeError):
            continue
    return tracks
