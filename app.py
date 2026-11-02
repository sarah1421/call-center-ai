"""Streamlit demo (bonus).

Run locally:   pip install streamlit pandas  &&  streamlit run app.py
Run on Colab:  notebooks/06_streamlit_demo_colab.ipynb

Tab 1 "Results Explorer" only reads results/final_results.json (no GPU or keys needed).
Tab 2 "Analyze a New Call" runs the full pipeline and needs a GPU, HF_TOKEN and GROQ_API_KEY.
"""
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.utils import load_json

st.set_page_config(page_title="Call Center AI", page_icon="📞", layout="wide")
SENTIMENT_ICON = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}


@st.cache_data
def load_results():
    return load_json(config.FINAL_JSON) if config.FINAL_JSON.exists() else []


@st.cache_resource(show_spinner="Loading Whisper, pyannote and the LLM (first time only)...")
def load_pipeline():
    from src.diarizer import SpeakerDiarizer
    from src.llm import GroqLLM
    from src.pipeline import CallCenterPipeline
    from src.role_labeler import RoleLabeler
    from src.sentiment import SentimentAnalyzer
    from src.summarizer import CallSummarizer
    from src.transcriber import WhisperTranscriber

    llm = GroqLLM()
    return CallCenterPipeline(WhisperTranscriber(), SpeakerDiarizer(config.get_secret("HF_TOKEN")),
                              RoleLabeler(llm), CallSummarizer(llm), SentimentAnalyzer(llm))


def show_call(turns, summary, sentiment, speakers, notes=None, audio=None):
    """Shared view: summary + sentiment cards, then the transcript as a chat."""
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Summary")
        st.markdown(f"**Customer issue:** {summary.get('customer_issue', '–')}")
        st.markdown(f"**Agent action:** {summary.get('agent_action', '–')}")
        st.markdown(f"**Outcome:** {summary.get('outcome', '–')}")
        st.info(summary.get("summary", "–"))
    with c2:
        st.subheader("Customer sentiment")
        overall = sentiment.get("overall_sentiment") or sentiment.get("sentiment") or "–"
        st.markdown(f"### {SENTIMENT_ICON.get(overall, '')} {overall}")
        start = sentiment.get("start_sentiment") or sentiment.get("sentiment_start")
        end = sentiment.get("end_sentiment") or sentiment.get("sentiment_end")
        trend = sentiment.get("trend") or sentiment.get("sentiment_trend")
        st.markdown(f"Start → End: **{start} → {end}** ({trend})")
        st.caption(sentiment.get("justification") or sentiment.get("sentiment_reason") or "")
        st.markdown(f"**Speakers:** {speakers}")
        if notes:
            st.warning(notes)
    if audio is not None:
        st.audio(audio)
    st.subheader("Transcript")
    for t in turns:
        avatar = "🎧" if t.get("role") == "Agent" else "🙋"
        with st.chat_message(t.get("role", "user"), avatar=avatar):
            flag = " ⚠️ *low confidence*" if t.get("low_confidence") else ""
            st.markdown(f"**{t.get('speaker_label', '')} · {t.get('role')}** `{t['start']:.1f}s`{flag}  \n{t['text']}")


st.title("📞 Call Center AI — Transcription, Summaries & Sentiment")
tab1, tab2, tab3 = st.tabs(["Results Explorer", "Analyze a New Call", "About"])

# ------------------------------------------------------------------ tab 1
with tab1:
    results = load_results()
    if not results:
        st.warning("No results yet. Run notebooks 01–05 first (they create results/final_results.json).")
    else:
        df = pd.DataFrame(results)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calls processed", len(df))
        wer_col = "wer_whisper_norm" if "wer_whisper_norm" in df and df["wer_whisper_norm"].notna().any() else "wer"
        m2.metric("Mean WER", f"{df[wer_col].mean():.1%}" if df[wer_col].notna().any() else "–")
        m3.metric("Mean role accuracy", f"{df['role_accuracy'].mean():.1%}" if df["role_accuracy"].notna().any() else "–")
        m4.metric("Negative calls", int((df["sentiment"] == "Negative").sum()))
        st.bar_chart(df["sentiment"].value_counts())

        f1, f2 = st.columns(2)
        sent_filter = f1.multiselect("Filter by sentiment", sorted(df["sentiment"].dropna().unique()))
        accent_filter = f2.multiselect("Filter by accent", sorted(df["accent"].dropna().unique()))
        view = df
        if sent_filter:
            view = view[view["sentiment"].isin(sent_filter)]
        if accent_filter:
            view = view[view["accent"].isin(accent_filter)]
        st.dataframe(view[["call_id", "accent", "duration_min", "sentiment", "sentiment_trend", "summary", wer_col]],
                     width="stretch", hide_index=True)

        if len(view):
            cid = st.selectbox("Open a call", view["call_id"])
            row = df[df["call_id"] == cid].iloc[0].to_dict()
            audio_file = config.AUDIO_DIR / f"{cid}.wav"
            show_call(row["turns"], row, row, row["speakers"], row.get("notes"),
                      str(audio_file) if audio_file.exists() else None)

# ------------------------------------------------------------------ tab 2
with tab2:
    st.write("Upload a two-speaker customer-service call (WAV / MP3). Needs a GPU, `HF_TOKEN` and `GROQ_API_KEY`.")
    upload = st.file_uploader("Call recording", type=["wav", "mp3", "m4a", "flac"])
    if upload and st.button("Analyze call", type="primary"):
        suffix = Path(upload.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload.getbuffer())
        try:
            pipe = load_pipeline()
            with st.status("Processing...", expanded=True) as status:
                st.write("Transcribing, separating speakers, summarizing, analysing sentiment")
                res = pipe.process_call("uploaded_call", tmp.name)
                status.update(label="Done", state="complete")
            show_call(res.turns, res.summary, res.sentiment,
                      ", ".join(f"{k} = {v}" for k, v in res.speaker_roles.items()),
                      "; ".join(res.notes[:-1]), upload)
        except Exception as e:
            st.error(f"Could not process the call: {e}")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.markdown("""
**Pipeline:** Whisper (speech-to-text) → pyannote (speaker diarization) → word-to-speaker alignment →
Agent/Customer labelling (LLM + keywords) → LLM summary (issue, action, outcome) → LLM customer sentiment (start vs. end).

**Data:** AppTek Call Center Dialogues (role-played English calls, CC BY-SA 4.0).
See the README for method, results and limitations.""")
    arch = config.PROJECT_ROOT / "docs" / "architecture.png"
    if arch.exists():
        st.image(str(arch), caption="High-level architecture")
