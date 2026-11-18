"""
CLI:  python -m src.cli "Cocteau Twins - Know Who You Are at Every Age" "Joy Division - Disorder" ...
      python -m src.cli --rate <album-id> up|down
      python -m src.cli --correct <album-id> --add "dream pop" --remove "rock"
"""
import argparse
import sys

from src.rag import logging_setup
from src.rag.feedback import Feedback
from src.rag.pipeline import AlbumRecommender, load_records


def main(argv=None) -> int:
    logging_setup.setup()
    ap = argparse.ArgumentParser(description="Album recommendations from 2-5 favorite songs.")
    ap.add_argument("songs", nargs="*", help="'Artist - Song' (2-5)")
    ap.add_argument("-k", type=int, default=12, help="number of albums (1-20)")
    ap.add_argument("--discover", action="store_true",
                    help="let the LLM suggest extra albums, verified on MusicBrainz (needs a key; ~30s the first time)")
    ap.add_argument("--rate", nargs=2, metavar=("ALBUM_ID", "up|down"))
    ap.add_argument("--correct", metavar="ALBUM_ID")
    ap.add_argument("--add", default="", help="comma-separated genre tags to add (with --correct)")
    ap.add_argument("--remove", default="", help="comma-separated genre tags to remove (with --correct)")
    args = ap.parse_args(argv)
    fb = Feedback()

    if args.rate:
        if args.rate[1] not in ("up", "down"):
            ap.error("--rate needs 'up' or 'down'")
        recorded = fb.vote(args.rate[0], args.rate[1] == "up", session_id="cli")
        print(f"Thanks for your feedback!" if recorded else "You already answered for this album.")
        return 0
    if args.correct:
        fb.correct(args.correct, args.add.split(","), args.remove.split(","))
        print(f"Recorded correction for {args.correct}")
        return 0

    try:
        recommender = AlbumRecommender(load_records(fb), fb)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    result = recommender.recommend(args.songs, args.k, ai_discovery=args.discover)
    if result.discovery_note:
        print(result.discovery_note)
    if result.message:
        print(result.message)
        return 2
    if result.unresolved:
        print(f"(skipped, no genre info: {', '.join(result.unresolved)})")
    print(f"{len(result.recommendations)} results ({result.elapsed_s:.2f} seconds)")
    if result.common_tags:
        print("Your songs have in common: " + "; ".join(f"{t} ({', '.join(a)})" for t, a in result.common_tags))
    for n, rec in enumerate(result.recommendations, 1):
        r = rec.record
        flags = f"  [{', '.join(r.flags)}]" if r.flags else ""
        flags += "  [AI-suggested, verified]" if r.origin == "llm" else ""
        print(f"\n{n:2}. {r.artist} - {r.title} ({r.year or '?'})  match {min(100, round(rec.score * 100))}%  confidence {round(r.confidence * 100)}%{flags}")
        print(f"    genres: {', '.join(sorted(r.genres, key=r.genres.get, reverse=True)[:4])}")
        print(f"    why: {rec.reason}")
        print(f"    {rec.blurb}")
        print(f"    id: {r.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
