"""
Source clients. API-only (no scraping): MusicBrainz, Wikipedia and Wikidata.

Every call goes through _get_json(): User-Agent, per-host rate limit, timeout, bounded retries.
Every result is cached in data/raw/<source>/<key>.json so ingest, tests and demos run offline.
Return contract: dict with data, {} when the source has no match (cached), None on transient failure (not cached).
"""
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import requests

log = logging.getLogger("rag.sources")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
TIMEOUT_S = 15
RETRIES = 3
MIN_INTERVAL_S = {"musicbrainz.org": 1.1, "en.wikipedia.org": 1.0, "www.wikidata.org": 1.0, "api.deezer.com": 0.2}
MB = "https://musicbrainz.org/ws/2"
_last_call: Dict[str, float] = {}


def _user_agent() -> str:
    return f"music-rag-class-project/0.1 ({os.environ.get('MB_CONTACT', 'no-contact-set')})"


def slug(*parts: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", " ".join(parts).lower()).strip("-")[:120]


def _get_json(url: str, params: Optional[dict] = None) -> Optional[dict]:
    host = url.split("/")[2]
    for attempt in range(1, RETRIES + 1):
        wait = MIN_INTERVAL_S.get(host, 0.5) - (time.monotonic() - _last_call.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()
        try:
            resp = requests.get(url, params=params, headers={"User-Agent": _user_agent()}, timeout=TIMEOUT_S)
            if resp.status_code == 404:
                return {}
            if resp.status_code in (429, 503):
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("GET %s failed (attempt %d/%d): %s", host, attempt, RETRIES, exc)
            time.sleep(2 ** attempt)
    return None


def _cached(source: str, key: str, fetch: Callable[[], Optional[dict]], refresh: bool = False,
            stale: Optional[Callable[[dict], bool]] = None) -> Optional[dict]:
    """File cache. `stale(cached)` marks old-format entries for re-fetch; if the re-fetch fails, the old entry is kept."""
    path = RAW_DIR / source / f"{key}.json"
    cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if cached is not None and not refresh and not (stale and stale(cached)):
        return cached
    result = fetch()
    if result is None:
        return cached
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


def _clean(text: str) -> str:
    return text.replace('"', " ").strip()


def musicbrainz_album(artist: str, title: str, refresh: bool = False) -> Optional[dict]:
    """Release-group genres (name -> vote count), community rating (0-5), year."""
    def fetch() -> Optional[dict]:
        found = _get_json(f"{MB}/release-group", {
            "query": f'releasegroup:"{_clean(title)}" AND artist:"{_clean(artist)}"', "fmt": "json", "limit": 1})
        if found is None:
            return None
        hits = found.get("release-groups", [])
        if not hits or hits[0].get("score", 0) < 80:
            return {}
        detail = _get_json(f"{MB}/release-group/{hits[0]['id']}", {"inc": "genres+ratings", "fmt": "json"})
        if not detail:
            return detail
        rating = detail.get("rating") or {}
        return {
            "mbid": detail["id"], "title": detail.get("title"),
            "year": (detail.get("first-release-date") or "")[:4] or None,
            "genres": {g["name"]: g.get("count", 1) for g in detail.get("genres", [])},
            "rating": rating.get("value"), "votes": rating.get("votes-count", 0),
        }
    return _cached("musicbrainz", slug(artist, title), fetch, refresh)


def musicbrainz_artist_genres(artist: str, refresh: bool = False) -> Optional[dict]:
    """Artist-level genres, used to resolve favorite songs whose artist is not in the corpus."""
    def fetch() -> Optional[dict]:
        found = _get_json(f"{MB}/artist", {"query": f'artist:"{_clean(artist)}"', "fmt": "json", "limit": 1})
        if found is None:
            return None
        hits = found.get("artists", [])
        if not hits or hits[0].get("score", 0) < 90:
            return {}
        detail = _get_json(f"{MB}/artist/{hits[0]['id']}", {"inc": "genres", "fmt": "json"})
        if not detail:
            return detail
        return {"name": detail.get("name"), "genres": {g["name"]: g.get("count", 1) for g in detail.get("genres", [])}}
    return _cached("musicbrainz_artist", slug(artist), fetch, refresh)


def split_wikipedia_text(text: str) -> Tuple[str, str]:
    """Plain-text article (sections marked '== Heading ==') -> (lead paragraph(s), reception section text, max 1200 chars)."""
    parts = re.split(r"^==\s*([^=].*?)\s*==\s*$", text, flags=re.M)      # [lead, head1, body1, head2, body2, ...]
    lead, reception = parts[0].strip(), ""
    for head, body in zip(parts[1::2], parts[2::2]):
        if re.search(r"reception|critical response|reviews", head, re.I):
            reception = re.sub(r"^=+.*?=+\s*$", "", body, flags=re.M).strip()
            break
    return lead, re.sub(r"\n{2,}", "\n", reception)[:1200]


def wikipedia_album(artist: str, title: str, refresh: bool = False) -> Optional[dict]:
    """
    Lead text, critical-reception text and Wikidata id, from one article. Rejected when the lead does not mention
    the artist (wrong page). Old cache entries without "reception" are re-fetched automatically.
    """
    def fetch() -> Optional[dict]:
        api = "https://en.wikipedia.org/w/api.php"
        found = _get_json(api, {"action": "query", "list": "search", "srsearch": f"{title} {artist} album",
                                "srlimit": 1, "format": "json"})
        if found is None:
            return None
        hits = found.get("query", {}).get("search", [])
        if not hits:
            return {}
        data = _get_json(api, {"action": "query", "prop": "extracts|pageprops", "titles": hits[0]["title"],
                               "explaintext": 1, "exsectionformat": "wiki", "ppprop": "wikibase_item",
                               "redirects": 1, "format": "json"})
        if not data:
            return data
        page = next(iter(data.get("query", {}).get("pages", {}).values()), {})
        lead, reception = split_wikipedia_text(page.get("extract", ""))
        if not lead or re.sub(r"^the ", "", artist.lower()) not in lead.lower():
            log.info("wikipedia page %r rejected for %s - %s (artist not in lead)", hits[0]["title"], artist, title)
            return {}
        return {"page": page.get("title"), "extract": lead, "reception": reception,
                "qid": page.get("pageprops", {}).get("wikibase_item")}
    return _cached("wikipedia", slug(artist, title), fetch, refresh, stale=lambda d: bool(d) and "reception" not in d)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", re.sub(r"^the ", "", text.lower())).strip()


def deezer_album_cover(artist: str, title: str, refresh: bool = False) -> Optional[dict]:
    """Album cover URL from Deezer's public search. Accepts only an exact artist match and a compatible title."""
    def fetch() -> Optional[dict]:
        data = _get_json("https://api.deezer.com/search/album",
                         {"q": f'artist:"{_clean(artist)}" album:"{_clean(title)}"', "limit": 5})
        if data is None:
            return None
        for album in data.get("data", []) if isinstance(data.get("data"), list) else []:
            try:
                same_artist = _norm(album["artist"]["name"]) == _norm(artist)
                found_title, want = _norm(album["title"]), _norm(title)
                cover = album.get("cover_big") or album.get("cover_medium") or ""
            except (KeyError, TypeError):
                continue
            if same_artist and (want in found_title or found_title in want) and cover.startswith("https://"):
                return {"cover": cover}
        return {}
    return _cached("deezer", slug(artist, title), fetch, refresh)


def wikidata_genres(qid: str, refresh: bool = False) -> Optional[dict]:
    """Genre labels (property P136) for a Wikidata item: an independent second genre source."""
    def fetch() -> Optional[dict]:
        api = "https://www.wikidata.org/w/api.php"
        ent = _get_json(api, {"action": "wbgetentities", "ids": qid, "props": "claims", "format": "json"})
        if ent is None:
            return None
        claims = (ent.get("entities", {}).get(qid, {}).get("claims", {})).get("P136", [])
        ids = [c["mainsnak"]["datavalue"]["value"]["id"] for c in claims if "datavalue" in c.get("mainsnak", {})]
        if not ids:
            return {"genres": []}
        labels = _get_json(api, {"action": "wbgetentities", "ids": "|".join(ids), "props": "labels",
                                 "languages": "en", "format": "json"})
        if labels is None:
            return None
        names = [e["labels"]["en"]["value"] for e in labels.get("entities", {}).values() if "en" in e.get("labels", {})]
        return {"genres": names}
    return _cached("wikidata", qid, fetch, refresh)
