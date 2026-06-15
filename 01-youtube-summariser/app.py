import re

import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
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

    This regex extracts the 11-character video ID from any of them.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Download the transcript from YouTube
# ─────────────────────────────────────────────────────────────────────────────
def get_transcript(video_id: str) -> str:
    """
    Uses youtube-transcript-api to fetch the video's captions.

    The API returns a list of dicts like:
      [{'text': 'Hello everyone', 'start': 0.0, 'duration': 1.5}, ...]

    We just join all the 'text' values into one big string.
    """
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join(entry["text"] for entry in transcript_list)


# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC: Summarize using Map-Reduce
# ─────────────────────────────────────────────────────────────────────────────
def summarize_transcript(transcript: str, api_key: str) -> str:
    """
    Map-Reduce Summarization — the solution to long text.

    The Problem:
        LLMs have a token limit (e.g. 4,096 tokens). A 1-hour video transcript
        can be 20,000+ words — way too long to send all at once.

    The Solution — Map-Reduce:
        Step 1 (MAP):    Split transcript into small chunks.
                         Summarize EACH chunk individually.
        Step 2 (REDUCE): Take all those chunk summaries and combine
                         them into ONE final summary.

    This is the same idea as MapReduce in distributed computing — break
    a big problem into small pieces, solve each piece, combine the results.
    """
    llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)

    # ── Step 1: Split the transcript into chunks ──────────────────────────────
    # chunk_size=4000 chars (~800 words) keeps each chunk well within token limits.
    # chunk_overlap=200 means consecutive chunks share 200 chars of context,
    # so we don't lose meaning at the boundaries between chunks.
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    chunks = splitter.split_text(transcript)

    # LangChain's summarize chain expects a list of Document objects
    docs = [Document(page_content=chunk) for chunk in chunks]

    # ── MAP prompt: what to do with each individual chunk ────────────────────
    map_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Summarise the following section of a YouTube video transcript.
Keep it concise — capture the key points only.

TRANSCRIPT SECTION:
{text}

SECTION SUMMARY:""",
    )

    # ── REDUCE prompt: how to combine all chunk summaries ────────────────────
    reduce_prompt = PromptTemplate(
        input_variables=["text"],
        template="""You have summaries of different sections of a YouTube video.
Combine them into one well-structured final summary with:

- **Overview** (2-3 sentences about what the video is about)
- **Key Points** (bullet list of the most important ideas)
- **Main Takeaway** (one sentence — the single most important thing)

SECTION SUMMARIES:
{text}

FINAL SUMMARY:""",
    )

    # ── LangChain's built-in map_reduce chain handles the loop for us ────────
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=reduce_prompt,
        verbose=False,
    )

    return chain.run(docs)


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
    st.markdown(
        """
    - Works best with videos that have auto-generated or manual captions
    - Longer videos (30+ min) take ~30 seconds to summarise
    - If a video has no captions, you'll see a TranscriptsDisabled error
    """
    )

# ── Main: URL input ───────────────────────────────────────────────────────────
url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

if st.button("✨ Summarise Video", type="primary", use_container_width=True):

    # Validation
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        st.stop()
    if not url:
        st.error("⚠️ Please paste a YouTube URL.")
        st.stop()

    # Extract video ID
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
            st.markdown(
                "**Common reasons:** The video has captions disabled, "
                "or it's a live stream with no transcript yet."
            )
            st.stop()

    word_count = len(transcript.split())
    chunk_count = max(1, word_count // 800)
    st.info(
        f"📄 Transcript loaded — **{word_count:,} words** across ~{chunk_count} chunks. "
        f"Summarising now..."
    )

    # Step 2: Summarize
    with st.spinner(f"🤖 Running map-reduce summarization (~{chunk_count * 3}s)..."):
        try:
            summary = summarize_transcript(transcript, api_key)
        except Exception as e:
            st.error(f"❌ Summarisation failed: {e}")
            st.stop()

    # Display results
    st.success("✅ Done!")
    st.markdown("---")
    st.markdown("## 📝 Summary")
    st.markdown(summary)

    # Show raw transcript (collapsed by default)
    with st.expander("📜 View Raw Transcript"):
        st.caption("Showing first 3,000 characters")
        st.text(transcript[:3000] + ("..." if len(transcript) > 3000 else ""))
