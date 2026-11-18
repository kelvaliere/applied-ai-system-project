"""RAG tests: no network, no API key. Run with: pytest"""
import json

import pytest

from src.rag import crosscheck, generate, sources
from src.rag.crosscheck import AlbumRecord
from src.rag.feedback import Feedback
from src.rag.pipeline import AlbumRecommender, parse_song


def rec(artist, title, genres, rating=80.0, confidence=1.0, flags=None):
    return AlbumRecord(id=f"{artist}-{title}".lower().replace(" ", "-"), artist=artist, title=title, year="1990",
                       genres=genres, consensus_tags=[], rating=rating, rating_votes=10, summary=f"{title} is an album by {artist}.",
                       sources=["musicbrainz", "wikidata"], confidence=confidence, flags=flags or [])


CORPUS = [
    rec("Cocteau Twins", "Heaven", {"dream pop": 1.0, "ethereal wave": 0.6}),
    rec("Joy Division", "Closer", {"post-punk": 1.0, "gothic rock": 0.5}),
    rec("Slowdive", "Souvlaki", {"shoegaze": 1.0, "dream pop": 0.9}),
    rec("The Cure", "Disintegration", {"gothic rock": 1.0, "post-punk": 0.6, "dream pop": 0.4}),
    rec("Siouxsie", "Juju", {"post-punk": 1.0, "gothic rock": 0.7}),
    rec("Beach House", "Teen Dream", {"dream pop": 1.0, "indie pop": 0.5}),
    rec("Kendrick", "TPAB", {"hip hop": 1.0, "jazz rap": 0.5}),
]


@pytest.fixture
def recommender(tmp_path):
    return AlbumRecommender(CORPUS, Feedback(tmp_path), artist_genres=lambda a: {}, describe=lambda items: (
        {i["id"]: "blurb" for i in items}, False))


def test_parse_song_accepts_dashes_and_rejects_garbage():
    assert parse_song("Cocteau Twins - Pearly-Dewdrops' Drops") == ("Cocteau Twins", "Pearly-Dewdrops' Drops")
    assert parse_song("Bowie – Heroes") == ("Bowie", "Heroes")
    for bad in ("", "no separator", " - song", "x" * 300):
        with pytest.raises(ValueError):
            parse_song(bad)


def test_recommends_related_albums_and_skips_input_artists(recommender):
    res = recommender.recommend(["Cocteau Twins - A", "Joy Division - B"], k=3)
    ids = [r.record.artist for r in res.recommendations]
    assert "Cocteau Twins" not in ids and "Joy Division" not in ids
    assert ids[0] == "The Cure"                      # bridges both songs
    assert "Kendrick" not in ids


def test_max_one_album_per_artist_and_reason_is_present(recommender):
    corpus = CORPUS + [rec("Slowdive", "Pygmalion", {"shoegaze": 1.0, "dream pop": 0.8})]
    r = AlbumRecommender(corpus, Feedback(recommender.feedback.votes_path.parent), lambda a: {}, lambda i: ({x["id"]: "b" for x in i}, False))
    res = r.recommend(["Cocteau Twins - A", "Joy Division - B"], k=10)
    artists = [x.record.artist for x in res.recommendations]
    assert len(artists) == len(set(artists))
    assert all(x.reason for x in res.recommendations)


def test_input_guardrails(recommender):
    assert "between" in recommender.recommend(["Cocteau Twins - A"]).message
    assert "Artist - Song" in recommender.recommend(["nonsense", "also nonsense"]).message


def test_unknown_artists_give_not_enough_data(recommender):
    res = recommender.recommend(["Nobody - X", "Nobody Else - Y"])
    assert res.message.startswith("Not enough data") and not res.recommendations


def test_votes_are_capped_across_many_sessions(tmp_path):
    fb = Feedback(tmp_path)
    for n in range(50):
        fb.vote("slowdive-souvlaki", up=False, session_id=f"s{n}")
    assert fb.adjustments()["slowdive-souvlaki"] == -0.15


def test_one_vote_per_session_per_album_cannot_be_spammed(tmp_path):
    fb = Feedback(tmp_path)
    assert fb.vote("a", up=True, session_id="s1") is True
    assert all(fb.vote("a", up=True, session_id="s1") is False for _ in range(20))
    assert fb.adjustments()["a"] == 0.05                          # one vote, not 21
    assert fb.vote("a", up=True, session_id="s2") is True         # another session counts separately


def test_result_reports_elapsed_time(recommender):
    assert recommender.recommend(["Cocteau Twins - A", "Joy Division - B"]).elapsed_s > 0
    assert recommender.recommend(["bad"]).elapsed_s > 0


def test_correction_overrides_without_touching_raw(tmp_path):
    fb = Feedback(tmp_path)
    fb.correct("kendrick-tpab", add_tags=["dream pop"], remove_tags=["hip hop"])
    fixed = fb.apply_corrections([rec("Kendrick", "TPAB", {"hip hop": 1.0})])[0]
    assert "dream pop" in fixed.genres and "hip hop" not in fixed.genres and "user_corrected" in fixed.flags


def test_crosscheck_consensus_and_flags():
    mb = {"genres": {"dream pop": 7, "gothic rock": 17}, "rating": 4.4, "votes": 29, "year": "1989"}
    both = crosscheck.merge("The Cure", "Disintegration", mb, {"extract": "x"}, {"genres": ["Gothic Rock"]})
    assert both.consensus_tags == ["gothic rock"] and both.confidence >= 0.9 and both.rating == 88.0
    single = crosscheck.merge("A", "B", mb, None, None)
    assert "single_genre_source" in single.flags and "low_confidence" in single.flags
    clash = crosscheck.merge("A", "B", mb, {"extract": "x"}, {"genres": ["jazz"]})
    assert "sources_disagree_on_genre" in clash.flags


def test_grounding_drops_invented_numbers_and_unknown_ids():
    items = [{"id": "a", "artist": "A", "title": "T", "year": "1989", "genres": ["x"], "rating": 88.0, "summary": "", "reason": "r"}]
    good = generate.validate({"a": "Released in 1989, rated 88.0."}, items)
    bad = generate.validate({"a": "Won 5 Grammys.", "zzz": "invented album"}, items)
    assert "a" in good and bad == {}


def test_describe_without_api_key_uses_templates(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(generate, "ENV_FILE", generate.Path("/nonexistent/.env"))
    items = [{"id": "a", "artist": "A", "title": "T", "year": "1", "genres": ["x"], "rating": None,
              "summary": "T is an album. More text.", "reason": "r"}]
    blurbs, used = generate.describe(items)
    assert blurbs == {"a": "T is an album."} and used is False


def test_describe_falls_back_when_llm_fails(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(generate, "_ask_llm", lambda system, payload, which: (_ for _ in ()).throw(ValueError("boom")))
    items = [{"id": "a", "artist": "A", "title": "T", "year": "1", "genres": ["x"], "rating": None, "summary": "", "reason": "r"}]
    blurbs, used = generate.describe(items)
    assert used is False and "T is a x album by A" in blurbs["a"]


def test_provider_prefers_gemini_and_reads_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "c")
    env = tmp_path / ".env"
    env.write_text("# comment\nGEMINI_API_KEY=\"g-key\"\n")
    monkeypatch.setattr(generate, "ENV_FILE", env)
    assert generate.provider() == "gemini"
    monkeypatch.delenv("GEMINI_API_KEY")


def test_gemini_response_is_parsed_and_validated(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key-123")
    payload = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": '{"a": "A fine 1989 record."}'}]}}]}

    class Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return payload

    seen = {}
    def fake_post(url, json, timeout, headers):
        seen["headers"], seen["url"] = headers, url
        return Resp()
    monkeypatch.setattr(generate.requests, "post", fake_post)
    items = [{"id": "a", "artist": "A", "title": "T", "year": "1989", "genres": ["x"], "rating": None, "summary": "", "reason": "r"}]
    blurbs, used = generate.describe(items)
    assert used is True and blurbs == {"a": "A fine 1989 record."}
    assert "secret-key-123" not in seen["url"] and seen["headers"]["x-goog-api-key"] == "secret-key-123"     # key only in the header


def test_unknown_artist_with_uncovered_genres_is_skipped_with_reason(tmp_path):
    kpop = lambda a: {"name": a, "genres": {"k-pop": 6, "hip hop": 1}}      # catalog only covers hip hop
    r = AlbumRecommender(CORPUS, Feedback(tmp_path), kpop, lambda i: ({x["id"]: "b" for x in i}, False))
    res = r.recommend(["Cocteau Twins - A", "Joy Division - B", "IU - Palette"])
    assert len(res.resolved) == 2 and "not in the catalog" in res.unresolved[0] and "k-pop" in res.unresolved[0]


def test_gemini_retries_on_high_demand_then_succeeds(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(generate.time, "sleep", lambda s: None)
    ok = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": '{"a": "Fine."}'}]}}]}
    calls = []

    class Resp:
        def __init__(self, code): self.status_code = code
        def raise_for_status(self): pass
        def json(self): return ok

    def fake_post(url, json, timeout, headers):
        calls.append(1)
        return Resp(503 if len(calls) < 3 else 200)
    monkeypatch.setattr(generate.requests, "post", fake_post)
    assert generate._ask_gemini("sys", [{"id": "a"}]) == {"a": "Fine."} and len(calls) == 3


# ---- discovery -------------------------------------------------------------------------------------------------
from src.rag import discover as discover_mod


def _fake_build(real):
    """build_fn stand-in: albums in `real` verify, everything else behaves like a hallucination (no MusicBrainz match)."""
    def build(artist, title, refresh=False, origin="seed", require_mb=False):
        r = rec(artist, title, {"k-pop": 1.0}) if (artist, title) in real else rec(artist, title, {})
        r.origin = origin
        return r, (artist, title) in real
    return build


def test_discovery_keeps_only_verified_albums_and_persists_them(tmp_path):
    store = tmp_path / "discovered.jsonl"
    suggestions = [("NewJeans", "Get Up"), ("Fake Band", "Imaginary Album"), ("IU", "Palette"), ("Slowdive", "Souvlaki")]
    new, note = discover_mod.discover(["IU - Palette"], CORPUS, store, lambda songs: suggestions,
                                      _fake_build({("NewJeans", "Get Up"), ("IU", "Palette")}))
    assert [r.title for r in new] == ["Get Up", "Palette"]            # hallucination dropped; Souvlaki is a duplicate of the catalog
    assert all(r.origin == "llm" for r in new) and "1 suggestion(s) rejected" in note
    assert len(discover_mod.load_discovered(store)) == 2


def test_discovery_respects_max_new_and_survives_llm_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(discover_mod, "MAX_NEW", 2)
    many = [(f"A{i}", "T") for i in range(8)]
    new, _ = discover_mod.discover(["x - y"], [], tmp_path / "d.jsonl", lambda s: many,
                                   _fake_build(set(many)))
    assert len(new) == 2

    def boom(songs):
        raise ValueError("model unavailable")
    new, note = discover_mod.discover(["x - y"], CORPUS, tmp_path / "e.jsonl", boom, _fake_build(set()))
    assert new == [] and "unavailable" in note and not (tmp_path / "e.jsonl").exists()


def test_suggest_cleans_llm_output(monkeypatch):
    monkeypatch.setattr(discover_mod.generate, "ask_json", lambda system, payload: [
        {"artist": " IU ", "album": "Palette"}, {"artist": "x"}, "junk", {"artist": "a" * 500, "album": "b"}])
    assert discover_mod.suggest(["IU - Palette"]) == [("IU", "Palette")]
    monkeypatch.setattr(discover_mod.generate, "ask_json", lambda system, payload: {"not": "a list"})
    with pytest.raises(ValueError):
        discover_mod.suggest(["x"])


def test_pipeline_adds_discovered_albums_before_ranking(tmp_path, monkeypatch):
    monkeypatch.setattr(discover_mod.generate, "provider", lambda: "gemini")
    kpop = rec("NewJeans", "Get Up", {"k-pop": 1.0, "dance-pop": 0.5}); kpop.origin = "llm"
    monkeypatch.setattr("src.rag.pipeline.discover.discover", lambda songs, known: ([kpop], "Added 1 new album(s)"))
    r = AlbumRecommender(CORPUS, Feedback(tmp_path), lambda a: {"name": a, "genres": {"k-pop": 5}},
                         lambda items: ({i["id"]: "b" for i in items}, False))
    res = r.recommend(["IU - Palette", "Taeyeon - I"], ai_discovery=True)
    assert res.discovery_note.startswith("Added 1") and len(res.resolved) == 2    # k-pop now in the catalog vocab
    assert res.recommendations[0].record.origin == "llm"


def test_discovery_stops_when_time_budget_is_spent(tmp_path, monkeypatch):
    monkeypatch.setattr(discover_mod, "MAX_SECONDS", 0)
    tried = []
    def build(artist, title, **kw):
        tried.append(artist)
        return rec(artist, title, {"x": 1.0}), True
    new, _ = discover_mod.discover(["x - y"], [], tmp_path / "d.jsonl", lambda s: [("A", "T"), ("B", "T")], build)
    assert new == [] and tried == []


# ---- reception text, covers, common tags -----------------------------------------------------------------------
def test_split_wikipedia_text_finds_reception_section():
    text = ("Lead sentence about the album.\n\n\n== Background ==\nStuff.\n\n== Critical reception ==\n"
            "Critics called it serene.\n\n=== Accolades ===\nIt topped lists.\n\n== Track listing ==\n1. Song")
    lead, reception = sources.split_wikipedia_text(text)
    assert lead == "Lead sentence about the album."
    assert "serene" in reception and "topped lists" in reception and "Accolades" not in reception and "Track" not in reception
    assert sources.split_wikipedia_text("Only a lead.")[1] == ""


def test_deezer_cover_requires_artist_and_title_match(monkeypatch, tmp_path):
    monkeypatch.setattr(sources, "RAW_DIR", tmp_path)
    data = {"data": [
        {"artist": {"name": "Somebody Else"}, "title": "Disintegration", "cover_big": "https://c/wrong.jpg"},
        {"artist": {"name": "The Cure"}, "title": "Disintegration (Deluxe)", "cover_big": "https://c/right.jpg"}]}
    monkeypatch.setattr(sources, "_get_json", lambda url, params=None: data)
    assert sources.deezer_album_cover("The Cure", "Disintegration") == {"cover": "https://c/right.jpg"}
    assert sources.deezer_album_cover("Nobody", "Nothing") == {}


def test_stale_wikipedia_cache_is_refetched_but_kept_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(sources, "RAW_DIR", tmp_path)
    (tmp_path / "wikipedia").mkdir()
    (tmp_path / "wikipedia" / "a-b.json").write_text('{"extract": "old lead", "qid": "Q1"}')     # old format, no "reception"
    monkeypatch.setattr(sources, "_get_json", lambda url, params=None: None)                      # network down
    assert sources.wikipedia_album("A", "B")["extract"] == "old lead"


def test_common_tags_lists_genres_shared_by_two_artists(recommender):
    res = recommender.recommend(["Cocteau Twins - A", "The Cure - B", "Joy Division - C"], k=3)
    shared = dict(res.common_tags)
    assert shared["dream pop"] == ["Cocteau Twins", "The Cure"]
    assert shared["gothic rock"] == ["Joy Division", "The Cure"] and shared["post-punk"] == ["Joy Division", "The Cure"]
    assert all(len(artists) >= 2 for artists in shared.values())
    assert "echoes what your songs have in common" in " ".join(r.reason for r in res.recommendations)


def test_template_blurb_prefers_critic_text():
    item = {"id": "a", "artist": "A", "title": "T", "genres": ["x"], "summary": "Lead one.",
            "reception": "Critics praised its serene mood. More."}
    assert generate.template_blurb(item) == "Critics praised its serene mood."
