"""
Favorite songs -> taste profile -> ranked album recommendations with reasons and descriptions.

Flow: parse/validate input -> resolve each song's artist to genre tags (corpus, else MusicBrainz) -> profile vector
-> score every album -> diversity selection -> describe -> log.

    score = 0.45*profile_similarity + 0.20*bridge + 0.20*rating + 0.10*confidence + feedback_adjustment
    bridge = share of the user's songs the album is related to (rewards albums that connect several tastes)
"""
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import crosscheck, discover, generate, sources
from .crosscheck import AlbumRecord, normalize_tag
from .feedback import Feedback
from .index import TagIndex

log = logging.getLogger("rag.pipeline")
ALBUMS_JSONL = Path(__file__).resolve().parents[2] / "data" / "albums.jsonl"

W_SIM, W_BRIDGE, W_RATING, W_CONF = 0.45, 0.20, 0.20, 0.10
RELATED_THRESHOLD = 0.25      # cosine at which an album counts as related to one song
MIN_BEST_SIMILARITY = 0.10    # below this we say "not enough data" instead of guessing
NEUTRAL_RATING = 0.5          # used when an album has no rating
MIN_GENRE_COVERAGE = 0.5      # share of an unknown artist's genre weight the catalog must cover
MIN_SONGS, MAX_SONGS, MAX_K, MAX_FIELD_CHARS = 2, 5, 20, 120


@dataclass
class Recommendation:
    record: AlbumRecord
    score: float
    cluster: int                 # index of the song this album matches best
    related_songs: List[int]
    shared_tags: List[str]
    reason: str
    blurb: str = ""


@dataclass
class Result:
    request_id: str
    recommendations: List[Recommendation] = field(default_factory=list)
    resolved: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    message: str = ""
    llm_used: bool = False
    elapsed_s: float = 0.0
    discovery_note: str = ""
    common_tags: List[Tuple[str, List[str]]] = field(default_factory=list)   # (genre, artists that share it)


def parse_song(raw: str) -> Tuple[str, str]:
    """'Artist - Song' (hyphen or dash) -> (artist, song). Raises ValueError with a user-facing message."""
    raw = raw.strip()
    if not raw or len(raw) > 2 * MAX_FIELD_CHARS:
        raise ValueError("Each song must be 'Artist - Song' (max 240 characters).")
    parts = re.split(r"\s+[-–—]\s+", raw, maxsplit=1)
    if len(parts) != 2 or not all(p.strip() for p in parts):
        raise ValueError(f"Could not read {raw!r}; use the form 'Artist - Song'.")
    return parts[0].strip(), parts[1].strip()


def _norm_artist(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", re.sub(r"^the ", "", name.lower())).strip()


def load_records(feedback: Feedback, path: Path = ALBUMS_JSONL) -> List[AlbumRecord]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.rag.ingest")
    records = crosscheck.load(path)
    known = {r.id for r in records}
    records += [r for r in discover.load_discovered() if r.id not in known]      # verified AI suggestions from past searches
    return feedback.apply_corrections(records)


class AlbumRecommender:
    def __init__(self, records: List[AlbumRecord], feedback: Feedback,
                 artist_genres: Callable[[str], Optional[dict]] = sources.musicbrainz_artist_genres,
                 describe: Callable[[List[dict]], Tuple[Dict[str, str], bool]] = generate.describe):
        self.records, self.feedback = records, feedback
        self.index = TagIndex(records)
        self._artist_genres, self._describe = artist_genres, describe

    def _add_records(self, new: List[AlbumRecord]) -> None:
        self.feedback.apply_corrections(new)
        self.records = self.records + new
        self.index = TagIndex(self.records)

    def _song_tags(self, artist: str) -> Tuple[Dict[str, float], str]:
        """
        Genre weights for an artist: mean over their albums in the corpus, else MusicBrainz artist genres.
        Returns (tags, why_unusable); why_unusable is "" when the tags can be used.
        """
        key = _norm_artist(artist)
        mine = [r for r in self.records if _norm_artist(r.artist) == key]
        if mine:
            tags: Dict[str, float] = {}
            for r in mine:
                for t, w in r.genres.items():
                    tags[t] = tags.get(t, 0.0) + w / len(mine)
            return tags, ("" if tags else "its albums have no genre data")
        remote = self._artist_genres(artist)
        if remote is None:
            return {}, "lookup failed (network?)"
        if not remote:
            return {}, "artist not found on MusicBrainz"
        genres = remote.get("genres") or {}
        if not genres:
            return {}, "MusicBrainz lists no genres for this artist"
        top = max(genres.values())
        tags = {normalize_tag(t): c / top for t, c in genres.items()}
        known = sum(w for t, w in tags.items() if t in self.index.vocab)
        if known / sum(tags.values()) < MIN_GENRE_COVERAGE:
            return {}, f"its genres ({', '.join(sorted(tags, key=tags.get, reverse=True)[:2])}) are not in the catalog yet"
        return tags, ""

    def recommend(self, raw_songs: List[str], k: int = 12, ai_discovery: bool = False) -> Result:
        """ai_discovery: let the LLM suggest extra albums; only those verified on MusicBrainz are added."""
        t0 = time.monotonic()
        result = self._recommend(raw_songs, k, ai_discovery)
        result.elapsed_s = time.monotonic() - t0
        log.info("req=%s done results=%d llm=%s seconds=%.2f", result.request_id, len(result.recommendations),
                 result.llm_used, result.elapsed_s)
        return result

    def _recommend(self, raw_songs: List[str], k: int, ai_discovery: bool) -> Result:
        result = Result(request_id=uuid.uuid4().hex[:8])
        k = max(1, min(int(k), MAX_K))
        try:
            if not MIN_SONGS <= len(raw_songs) <= MAX_SONGS:
                raise ValueError(f"Enter between {MIN_SONGS} and {MAX_SONGS} songs.")
            parsed = [parse_song(s) for s in raw_songs]
        except ValueError as exc:
            result.message = str(exc)
            log.info("req=%s rejected input: %s", result.request_id, exc)
            return result

        if ai_discovery:
            if generate.provider():
                new, result.discovery_note = discover.discover([f"{a} - {t}" for a, t in parsed], self.records)
                if new:
                    self._add_records(new)
            else:
                result.discovery_note = "AI discovery needs GEMINI_API_KEY or ANTHROPIC_API_KEY in .env."

        song_vecs, labels, song_tags = [], [], []
        for artist, song in parsed:
            label = f"{artist} - {song}"
            tags, why = self._song_tags(artist)
            vec = self.index.vector(tags)
            if why or not vec.any():
                result.unresolved.append(f"{label} ({why or 'no genres in the catalog'})")
                continue
            song_vecs.append(vec)
            labels.append(label)
            song_tags.append((artist, tags))
        result.resolved = labels
        if len(song_vecs) < MIN_SONGS:
            result.message = ("Not enough data: I could only find genre info for "
                              f"{len(song_vecs)} song(s). Skipped: {'; '.join(result.unresolved)}.")
            log.info("req=%s unresolved=%s", result.request_id, result.unresolved)
            return result

        result.common_tags = self._common_tags(song_tags)
        common = {t for t, _ in result.common_tags}
        profile = song_vecs[0].copy()
        for v in song_vecs[1:]:
            profile += v
        profile /= np.linalg.norm(profile)
        sims = self.index.similarity(profile)
        per_song = np.stack([self.index.similarity(v) for v in song_vecs])        # songs x albums
        related = per_song >= RELATED_THRESHOLD
        adjust = self.feedback.adjustments()
        input_artists = {_norm_artist(a) for a, _ in parsed}

        if float(sims.max()) < MIN_BEST_SIMILARITY:
            result.message = "Not enough data: nothing in the catalog is close to this taste yet."
            log.info("req=%s weak retrieval best=%.3f", result.request_id, float(sims.max()))
            return result

        scored: List[Recommendation] = []
        for i, rec in enumerate(self.records):
            if _norm_artist(rec.artist) in input_artists:
                continue                                   # discovery: skip artists the user already named
            bridge = float(related[:, i].sum()) / len(song_vecs)
            rating = (rec.rating / 100) if rec.rating is not None else NEUTRAL_RATING
            score = (W_SIM * float(sims[i]) + W_BRIDGE * bridge + W_RATING * rating
                     + W_CONF * rec.confidence + adjust.get(rec.id, 0.0))
            cluster = int(per_song[:, i].argmax())
            rel = [int(j) for j in np.flatnonzero(related[:, i])] or [cluster]
            shared = self._shared_tags(rec, [song_vecs[j] for j in rel])
            scored.append(Recommendation(rec, round(score, 4), cluster, rel, shared,
                                         self._reason(rec, labels, rel, shared, common)))

        chosen = self._select(sorted(scored, key=lambda r: r.score, reverse=True), k, len(song_vecs))
        blurbs, result.llm_used = self._describe(self._items(chosen))
        for rec in chosen:
            rec.blurb = blurbs[rec.record.id]
        result.recommendations = chosen
        log.info("req=%s songs=%d resolved=%d k=%d top=%s", result.request_id, len(parsed), len(labels),
                 len(chosen), [r.record.id for r in chosen[:3]])
        return result

    def _shared_tags(self, rec: AlbumRecord, vecs: List[np.ndarray]) -> List[str]:
        inv = {i: t for t, i in self.index.vocab.items()}
        want = np.sum(vecs, axis=0)
        ranked = sorted(rec.genres, key=lambda t: want[self.index.vocab[t]] * rec.genres[t], reverse=True)
        return [t for t in ranked if want[self.index.vocab[t]] > 0][:3]

    @staticmethod
    def _common_tags(song_tags: List[Tuple[str, Dict[str, float]]], top: int = 5) -> List[Tuple[str, List[str]]]:
        """Genres that at least two different artists among the user's songs share (their strongest tags only)."""
        by_tag: Dict[str, Dict[str, float]] = {}
        for artist, tags in song_tags:
            peak = max(tags.values(), default=0.0)
            for tag, w in tags.items():
                if peak and w >= 0.3 * peak:
                    by_tag.setdefault(tag, {})[artist] = w
        shared = [(tag, artists) for tag, artists in by_tag.items() if len(artists) >= 2]
        shared.sort(key=lambda item: (len(item[1]), sum(item[1].values())), reverse=True)
        return [(tag, sorted(artists)) for tag, artists in shared[:top]]

    @staticmethod
    def _reason(rec: AlbumRecord, labels: List[str], related: List[int], shared: List[str], common: set) -> str:
        songs = " + ".join(labels[j].split(" - ")[0] for j in related)
        tags = ", ".join(shared) if shared else ", ".join(list(rec.genres)[:3])
        text = f"Fits your taste in {songs} through shared genres: {tags}."
        echoes = [t for t in rec.genres if t in common]
        if echoes:
            text += f" It also echoes what your songs have in common ({', '.join(echoes[:2])})."
        return text

    @staticmethod
    def _select(ranked: List[Recommendation], k: int, n_clusters: int) -> List[Recommendation]:
        """Max one album per artist; reserve slots so each song's taste is represented; fill by score."""
        per_cluster = 2 if k >= 2 * n_clusters else 1
        chosen, artists = [], set()

        def take(rec: Recommendation) -> bool:
            if rec.record.artist in artists or rec in chosen:
                return False
            chosen.append(rec)
            artists.add(rec.record.artist)
            return True

        for c in range(n_clusters):
            taken = 0
            for rec in ranked:
                if taken >= per_cluster or len(chosen) >= k:
                    break
                if rec.cluster == c and take(rec):
                    taken += 1
        for rec in ranked:
            if len(chosen) >= k:
                break
            take(rec)
        return sorted(chosen, key=lambda r: r.score, reverse=True)

    @staticmethod
    def _items(recs: List[Recommendation]) -> List[dict]:
        return [{"id": r.record.id, "artist": r.record.artist, "title": r.record.title, "year": r.record.year,
                 "genres": sorted(r.record.genres, key=r.record.genres.get, reverse=True)[:5],
                 "rating": r.record.rating, "summary": r.record.summary[:400], "reception": r.record.reception[:900]}
                for r in recs]
