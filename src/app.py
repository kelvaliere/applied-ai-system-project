"""Streamlit UI:  streamlit run src/app.py   (run from the project root)"""
import html
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.rag import generate, logging_setup, tracks
from src.rag.feedback import Feedback
from src.rag.pipeline import MAX_K, MAX_SONGS, MIN_SONGS, AlbumRecommender, load_records

logging_setup.setup()
st.set_page_config(page_title="Album recommender", page_icon="🎵", layout="centered")
st.title("🎵 Album recommender")
st.caption("Pick 2-5 songs you love. Data: MusicBrainz, Wikipedia, Wikidata (API only, cross-checked). "
           "Track search and previews: Deezer.")

feedback = Feedback()
session_id = st.session_state.setdefault("sid", uuid.uuid4().hex[:12])
answers = st.session_state.setdefault("answers", {})      # album_id -> "yes" | "no" (one answer per album per session)
songs = st.session_state.setdefault("songs", [])          # picked songs: {"label", "detail", "cover", "preview"}


@st.cache_data(ttl=600, show_spinner=False)
def search(query: str):
    return tracks.search_tracks(query)


def add_song(label: str, detail: str = "", cover: str = "", preview: str = "") -> None:
    if len(songs) < MAX_SONGS and label not in {s["label"] for s in songs}:
        songs.append({"label": label, "detail": detail, "cover": cover, "preview": preview})
    st.session_state["q"] = ""                             # clear the search box after adding


def remove_song(index: int) -> None:
    songs.pop(index)


def answer(album_id: str, yes: bool) -> None:
    """Button callback: records feedback once and shows a pop-up. Does not re-rank the page you are looking at."""
    if album_id in answers:
        return
    answers[album_id] = "yes" if yes else "no"
    feedback.vote(album_id, yes, session_id)
    st.toast("Thanks for your feedback!", icon="🙏")


# ---- 1. pick songs ------------------------------------------------------------------------------------------------
st.subheader("1. Pick your songs")
query = st.text_input("Search for a track (artist or title, then press Enter)", key="q", max_chars=tracks.MAX_QUERY)
full = len(songs) >= MAX_SONGS
if query.strip():
    found = search(query.strip())
    if not found:
        st.caption("No tracks found.")
    for i, t in enumerate(found):
        row = st.columns([1, 6, 2], vertical_alignment="center")
        if t.cover:
            row[0].image(t.cover, width=48)
        row[1].markdown(f"**{html.escape(t.title)}**  \n{html.escape(t.artist)} · {html.escape(t.album)}",
                        unsafe_allow_html=True)
        row[2].button("Add", key=f"add-{i}", on_click=add_song, args=(t.label, t.album, t.cover, t.preview),
                      disabled=full or t.label in {s["label"] for s in songs})
        if t.preview:
            st.audio(t.preview)
    if " - " in query:
        st.button(f'Add "{query.strip()}" as typed', on_click=add_song, args=(query.strip(),), disabled=full)

st.markdown(f"**Your songs ({len(songs)}/{MAX_SONGS})**")
if not songs:
    st.caption("Nothing yet. Search above and click Add.")
for i, s in enumerate(songs):
    row = st.columns([1, 7, 1], vertical_alignment="center")
    if s["cover"]:
        row[0].image(s["cover"], width=48)
    row[1].markdown(html.escape(s["label"]) + (f"  \n<small>{html.escape(s['detail'])}</small>" if s["detail"] else ""),
                    unsafe_allow_html=True)
    row[2].button("✕", key=f"rm-{i}", on_click=remove_song, args=(i,), help="Remove")
    if s["preview"]:
        st.audio(s["preview"])

# ---- 2. options + run ---------------------------------------------------------------------------------------------
st.subheader("2. Get albums")
k = st.number_input("How many albums do you want?", min_value=1, max_value=MAX_K, value=10, step=1)
ai_ok = generate.provider() is not None
ai_discovery = st.checkbox("Let AI suggest extra albums (verified on MusicBrainz; ~30s the first time)",
                           value=ai_ok, disabled=not ai_ok,
                           help=None if ai_ok else "Set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env to enable.")
go = st.button("Get recommendations", type="primary", disabled=len(songs) < MIN_SONGS)
if len(songs) < MIN_SONGS:
    st.caption(f"Add at least {MIN_SONGS} songs.")

if go:
    # Computed once per search and kept in session state, so clicking feedback never reshuffles the results.
    try:
        recommender = AlbumRecommender(load_records(feedback), feedback)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    with st.spinner("Finding albums" + (" (asking AI for candidates and verifying them, first time takes a while)..."
                                        if ai_discovery else "...")):
        st.session_state["result"] = recommender.recommend([s["label"] for s in songs], int(k), ai_discovery)
    answers.clear()

# ---- 3. results ---------------------------------------------------------------------------------------------------
def pct(value: float) -> str:
    return f"{max(0, min(100, round(value * 100)))}%"


def render_card(rec) -> None:
    r = rec.record
    with st.container(border=True):
        left, right = st.columns([1, 2.3], gap="medium")

        with left:                                            # album art, then the numbers
            if r.cover:
                st.image(r.cover, use_container_width=True)
            else:
                st.markdown("<div style='font-size:72px;text-align:center;opacity:.35'>🎵</div>", unsafe_allow_html=True)
            st.metric("Match", pct(rec.score), help="How well this album fits your songs (similarity, bridge, rating, "
                                                    "data confidence, feedback).")
            st.metric("Confidence", pct(r.confidence), help="How much our data sources agree about this album.")
            st.metric("Rating", f"{r.rating:.0f}/100" if r.rating is not None else "n/a", help="MusicBrainz community rating.")

        with right:                                           # title, artist/year, genres, description, why
            st.markdown(f"## {html.escape(r.title)}\n<small>{html.escape(r.artist)} · {r.year or '?'}</small>",
                        unsafe_allow_html=True)
            genres = sorted(r.genres, key=r.genres.get, reverse=True)[:5]
            if genres:
                st.markdown(f"<small>**Genres:** {html.escape(', '.join(genres))}</small>", unsafe_allow_html=True)
            if r.origin == "llm":
                st.caption("✨ Suggested by AI, then verified on MusicBrainz")
            st.write(rec.blurb)
            st.markdown(f"**Why:** {rec.reason}")
            if r.flags:
                st.warning("Check this data: " + ", ".join(r.flags))
            st.caption("Sources: " + ", ".join(r.sources))
            given = answers.get(r.id)
            if given:
                st.caption(f"✓ You answered {given}. Thanks for your feedback!")
            else:
                st.write("Was this a good recommendation?")
                yes, no, _ = st.columns([1, 1, 4])
                yes.button("Yes", key=f"yes-{r.id}", on_click=answer, args=(r.id, True))
                no.button("No", key=f"no-{r.id}", on_click=answer, args=(r.id, False))
            with st.expander("Correct the genres"):
                add = st.text_input("Add genres (comma-separated)", key=f"add-{r.id}")
                rem = st.text_input("Remove genres (comma-separated)", key=f"rem-{r.id}")
                if st.button("Save correction", key=f"fix-{r.id}"):
                    feedback.correct(r.id, add.split(","), rem.split(","))
                    st.toast("Thanks, correction saved. It applies to your next search.", icon="✏️")


result = st.session_state.get("result")
if result is not None and not hasattr(result, "common_tags"):      # saved by an older version of the app: discard it
    del st.session_state["result"]
    result = None
    st.info("The app was updated. Please search again.")
if result is not None:
    st.divider()
    if result.message:
        st.warning(result.message)
        st.stop()
    st.caption(f"About {len(result.recommendations)} results loaded ({result.elapsed_s:.2f} seconds)")
    if result.discovery_note:
        st.info(result.discovery_note)
    if result.unresolved:
        st.info("Skipped: " + "; ".join(result.unresolved))
    if result.common_tags:
        st.markdown("**What your songs have in common:** " +
                    " · ".join(f"{html.escape(t)} ({html.escape(', '.join(a))})" for t, a in result.common_tags))
    else:
        st.markdown("**Your songs cover different styles**, so these results blend several tastes.")
    if not result.llm_used:
        st.caption("Descriptions are template-based (no API key set, or the model output failed validation).")
    for rec in result.recommendations:
        render_card(rec)
