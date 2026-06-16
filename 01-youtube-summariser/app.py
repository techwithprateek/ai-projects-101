import streamlit as st

from core import get_transcript, get_video_id, summarize_transcript

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="YouTube Summariser", page_icon="🎬")
st.title("🎬 YouTube Video Summariser")
st.caption("Paste any YouTube URL and get a structured summary in seconds.")

# ── Sidebar — API key ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    st.markdown("Get your key at [platform.openai.com](https://platform.openai.com/api-keys)")

# ── Main UI ──────────────────────────────────────────────────────────────────
url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("✨ Summarise", use_container_width=True):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    if not url:
        st.error("Please enter a YouTube URL.")
        st.stop()

    video_id = get_video_id(url)
    if not video_id:
        st.error("Could not extract a video ID. Please check the URL.")
        st.stop()

    with st.spinner("Fetching transcript..."):
        try:
            transcript = get_transcript(video_id)
        except Exception as e:
            st.error(f"Could not fetch transcript: {e}")
            st.stop()

    st.info(f"📄 Transcript fetched — {len(transcript):,} characters. Summarising...")

    with st.spinner("Summarising with AI..."):
        try:
            summary = summarize_transcript(transcript, api_key)
        except Exception as e:
            st.error(f"Summarisation failed: {e}")
            st.stop()

    st.success("Done!")
    st.markdown("---")
    st.markdown(summary)
