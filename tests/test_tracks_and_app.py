"""Track search parsing and the Streamlit pick-songs flow. No network, no API keys."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.rag import generate, sources, tracks
from src.rag.tracks import Track

APP = Path(__file__).resolve().parents[1] / "src" / "app.py"


def test_search_tracks_parses_and_skips_bad_items(monkeypatch):
    payload = {"data": [
        {"title": "Palette", "artist": {"name": "IU"}, "album": {"title": "Palette", "cover_small": "https://c/x.jpg"},
         "preview": "https://p/x.mp3"},
        {"title": "No album"},                                                           # malformed -> skipped
        {"title": "T", "artist": {"name": "A"}, "album": {"title": "B", "cover_small": "http://insecure"}, "preview": None},
    ]}
    monkeypatch.setattr(sources, "_get_json", lambda url, params=None: payload)
    found = tracks.search_tracks("iu palette")
    assert [t.label for t in found] == ["IU - Palette", "A - T"]
    assert found[0].preview == "https://p/x.mp3" and found[1].cover == "" and found[1].preview == ""


@pytest.mark.parametrize("response", [None, {}, {"error": {"code": 4}}, {"data": "oops"}])
def test_search_tracks_degrades_to_empty(monkeypatch, response):
    monkeypatch.setattr(sources, "_get_json", lambda url, params=None: response)
    assert tracks.search_tracks("anything") == []
    assert tracks.search_tracks("x") == []                                               # too short, no request needed


def test_app_starts_empty_and_lets_user_pick_songs_and_count(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(generate, "ENV_FILE", tmp_path / "no.env")
    monkeypatch.setattr(tracks, "search_tracks", lambda q: [
        Track("Disorder", "Joy Division", "Unknown Pleasures", "", ""),
        Track("Know Who You Are at Every Age", "Cocteau Twins", "Four-Calendar Café", "", "")])
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert [t.value for t in at.text_input if t.key == "q"] == [""]                       # no default songs
    assert next(b for b in at.button if b.label == "Get recommendations").disabled        # needs 2 songs
    assert at.number_input[0].value == 10 and at.number_input[0].max == 20

    at.text_input(key="q").set_value("joy").run()
    next(b for b in at.button if b.key == "add-0").click().run()
    at.text_input(key="q").set_value("cocteau").run()
    next(b for b in at.button if b.key == "add-1").click().run()
    assert [s["label"] for s in at.session_state["songs"]] == ["Joy Division - Disorder",
                                                                "Cocteau Twins - Know Who You Are at Every Age"]
    assert at.text_input(key="q").value == ""                                             # box cleared after adding
    go = next(b for b in at.button if b.label == "Get recommendations")
    assert not go.disabled
    at.number_input[0].set_value(3).run()
    go.click().run()
    assert not at.exception
    assert len([m for m in at.markdown if m.value.startswith("## ")]) == 3                # chose 3 albums, got 3
    match = [m.value for m in at.metric if m.label == "Match"]
    assert len(match) == 3 and all(v.endswith("%") for v in match)                        # score shown as a percentage
    assert any("What your songs have in common" in m.value or "different styles" in m.value for m in at.markdown)


def test_app_discards_a_result_saved_by_an_older_version(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(generate, "ENV_FILE", tmp_path / "no.env")

    class OldResult:                       # looks like a Result from before common_tags existed
        message, recommendations, elapsed_s = "", [], 0.0

    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["result"] = OldResult()
    at.run()
    assert not at.exception
    assert any("search again" in i.value for i in at.info) and "result" not in at.session_state
