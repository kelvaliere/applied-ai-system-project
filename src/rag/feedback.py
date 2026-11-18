"""
Append-only user feedback. Raw data is never mutated.
  data/feedback.jsonl    {"ts", "album_id", "vote": 1 | -1, "session"}   (one vote per session per album)
  data/corrections.jsonl {"ts", "album_id", "add_tags": [...], "remove_tags": [...]}
Votes become a small, capped score adjustment; corrections are applied over records at load time.
"""
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .crosscheck import AlbumRecord, normalize_tag

log = logging.getLogger("rag.feedback")
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
VOTE_STEP = 0.05
VOTE_CAP = 0.15          # a handful of votes cannot dominate the ranking
USER_TAG_WEIGHT = 1.0


class Feedback:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.votes_path = data_dir / "feedback.jsonl"
        self.corrections_path = data_dir / "corrections.jsonl"

    @staticmethod
    def _append(path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **row}) + "\n")

    @staticmethod
    def _read(path: Path) -> List[dict]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("skipping corrupt line in %s", path.name)
        return rows

    def vote(self, album_id: str, up: bool, session_id: str = "anon") -> bool:
        """Record yes/no feedback once per (session, album). Returns False when this session already answered."""
        if any(r.get("album_id") == album_id and r.get("session") == session_id for r in self._read(self.votes_path)):
            log.info("duplicate vote ignored album=%s session=%s", album_id, session_id)
            return False
        self._append(self.votes_path, {"album_id": album_id, "vote": 1 if up else -1, "session": session_id})
        log.info("vote album=%s up=%s session=%s", album_id, up, session_id)
        return True

    def correct(self, album_id: str, add_tags: List[str], remove_tags: List[str]) -> None:
        add = [normalize_tag(t) for t in add_tags if t.strip()]
        rem = [normalize_tag(t) for t in remove_tags if t.strip()]
        if not add and not rem:
            return
        self._append(self.corrections_path, {"album_id": album_id, "add_tags": add, "remove_tags": rem})
        log.info("correction album=%s add=%s remove=%s", album_id, add, rem)

    def adjustments(self) -> Dict[str, float]:
        net: Dict[str, int] = defaultdict(int)
        for row in self._read(self.votes_path):
            net[row["album_id"]] += row["vote"]
        return {a: max(-VOTE_CAP, min(VOTE_CAP, n * VOTE_STEP)) for a, n in net.items()}

    def apply_corrections(self, records: List[AlbumRecord]) -> List[AlbumRecord]:
        by_id = {r.id: r for r in records}
        for row in self._read(self.corrections_path):
            rec = by_id.get(row["album_id"])
            if rec is None:
                continue
            for tag in row.get("remove_tags", []):
                rec.genres.pop(tag, None)
            for tag in row.get("add_tags", []):
                rec.genres[tag] = USER_TAG_WEIGHT
            if "user_corrected" not in rec.flags:
                rec.flags.append("user_corrected")
        return records
