import re

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Extract the video ID from a YouTube URL
# ─────────────────────────────────────────────────────────────────────────────
def get_video_id(url: str) -> str | None:
    """
    YouTube URLs come in several formats:
      - https://www.youtube.com/watch?v=dQw4w9WgXcQ
      - https://youtu.be/dQw4w9WgXcQ
      - https://www.youtube.com/embed/dQw4w9WgXcQ

    We match YouTube-specific patterns only (not just any 11-char path segment)
    to avoid false positives on non-YouTube URLs.
    """
    patterns = [
        r"youtube\.com/watch\?v=([0-9A-Za-z_-]{11})",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"youtube\.com/embed/([0-9A-Za-z_-]{11})",
        r"youtube\.com/v/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Download the transcript from YouTube
# ─────────────────────────────────────────────────────────────────────────────
def get_transcript(video_id: str) -> str:
    """
    Uses youtube-transcript-api (v1.x) to fetch the video captions.

    v1.x uses an instance-based API:
        api = YouTubeTranscriptApi()
        result = api.fetch(video_id)   ← returns a FetchedTranscript

    Each snippet in the result has a .text attribute.
    We join them all into one big string.
    """
    api = YouTubeTranscriptApi()
    snippets = api.fetch(video_id)
    return " ".join(snippet.text for snippet in snippets)


# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC: Summarize using Map-Reduce (built with LCEL)
# ─────────────────────────────────────────────────────────────────────────────
def summarize_transcript(transcript: str, api_key: str) -> str:
    """
    Map-Reduce Summarization — the solution to long text.

    The Problem:
        LLMs have a token limit (~4,000 tokens for GPT-3.5). A 1-hour video
        transcript can be 20,000+ words — way too long to send all at once.

    The Solution — Map-Reduce:
        Step 1 (MAP):    Split transcript into small chunks.
                         Summarize EACH chunk individually.
        Step 2 (REDUCE): Take all chunk summaries → combine into ONE final summary.

    We build this with LCEL (LangChain Expression Language):
        chain = prompt | llm | parser
    Each piece is a "runnable" you can pipe together like Unix commands.
    """
    llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)
    parser = StrOutputParser()  # converts AIMessage → plain string

    # ── Step 1: Split the transcript into chunks ──────────────────────────────
    # chunk_size=4000 chars (~800 words) keeps each chunk within token limits.
    # chunk_overlap=200 means consecutive chunks share 200 chars of context
    # so we don't lose meaning at chunk boundaries.
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    chunks = splitter.split_text(transcript)

    # ── MAP: summarize each chunk individually ────────────────────────────────
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that summarizes text concisely."),
        ("human", """Summarise the following section of a YouTube video transcript.
Be concise — capture the key points only.

TRANSCRIPT SECTION:
{chunk}

SECTION SUMMARY:"""),
    ])
    map_chain = map_prompt | llm | parser

    chunk_summaries = []
    for chunk in chunks:
        summary = map_chain.invoke({"chunk": chunk})
        chunk_summaries.append(summary)

    # ── REDUCE: combine all chunk summaries into one final summary ────────────
    combined = "\n\n---\n\n".join(chunk_summaries)

    reduce_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that creates clear, structured summaries."),
        ("human", """You have summaries of different sections of a YouTube video.
Combine them into one well-structured final summary with:

- **Overview** (2-3 sentences about what the video is about)
- **Key Points** (bullet list of the most important ideas)
- **Main Takeaway** (one sentence — the single most important thing)

SECTION SUMMARIES:
{summaries}

FINAL SUMMARY:"""),
    ])
    reduce_chain = reduce_prompt | llm | parser
    return reduce_chain.invoke({"summaries": combined})


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="YouTube Summariser", page_icon="🎬", layout="centered")

st.title("🎬 YouTube Video Summariser")
st.markdown("Paste any YouTube link and get a clean, structured summary — **no watching required.**")

# ── Sidebar: API key input ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your key is never stored — it only lives in this session.",
    )
    st.markdown("[Get your API key →](https://platform.openai.com/api-keys)")
    st.markdown("---")
    st.markdown("**💡 Tips**")
    st.markdown("""
    - Works best with videos that have auto-generated or manual captions
    - Longer videos (30+ min) take ~30 seconds to summarise
    - If a video has no captions, you'll see a TranscriptsDisabled error
    """)

# ── Main: URL input ───────────────────────────────────────────────────────────
url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("✨ Summarise Video", type="primary", use_container_width=True):

    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        st.stop()
    if not url:
        st.error("⚠️ Please paste a YouTube URL.")
        st.stop()

    video_id = get_video_id(url)
    if not video_id:
        st.error("⚠️ Couldn't find a valid YouTube video ID in that URL.")
        st.stop()

    # Step 1: Fetch transcript
    with st.spinner("📥 Fetching transcript from YouTube..."):
        try:
            transcript = get_transcript(video_id)
        except Exception as e:
            st.error(f"❌ Couldn't fetch transcript: {e}")
            st.markdown("**Common reasons:** Captions are disabled, or it's a live stream.")
            st.stop()

    word_count = len(transcript.split())
    chunk_count = max(1, len(transcript) // 4000)
    st.info(f"📄 Transcript loaded — **{word_count:,} words** across ~{chunk_count} chunks. Summarising...")

    # Step 2: Summarize
    with st.spinner(f"🤖 Running map-reduce summarization (~{chunk_count * 3}s)..."):
        try:
            summary = summarize_transcript(transcript, api_key)
        except Exception as e:
            st.error(f"❌ Summarisation failed: {e}")
            st.stop()

    st.success("✅ Done!")
    st.markdown("---")
    st.markdown("## 📝 Summary")
    st.markdown(summary)

    with st.expander("📜 View Raw Transcript"):
        st.caption("Showing first 3,000 characters")
        st.text(transcript[:3000] + ("..." if len(transcript) > 3000 else ""))
