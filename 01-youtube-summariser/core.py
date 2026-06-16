import re
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

from config import CHUNK_OVERLAP, CHUNK_SIZE, MODEL, TEMPERATURE

# Load prompt templates from the prompts/ folder at import time
_PROMPTS = Path(__file__).parent / "prompts"
MAP_PROMPT_TEXT    = (_PROMPTS / "map.txt").read_text()
REDUCE_PROMPT_TEXT = (_PROMPTS / "reduce.txt").read_text()


def get_video_id(url: str) -> str | None:
    """Extract the 11-character video ID from any YouTube URL format."""
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


def get_transcript(video_id: str) -> str:
    """Fetch and return the full transcript of a YouTube video as a single string."""
    api = YouTubeTranscriptApi()
    snippets = api.fetch(video_id)
    return " ".join(snippet.text for snippet in snippets)


def summarize_transcript(transcript: str, api_key: str) -> str:
    """
    Summarize a transcript using map-reduce over chunks.

    MAP:    Split transcript into chunks → summarize each one individually.
    REDUCE: Combine all chunk summaries → produce one final structured summary.
    """
    llm    = ChatOpenAI(api_key=api_key, model=MODEL, temperature=TEMPERATURE)
    parser = StrOutputParser()

    # Split into chunks that fit within the model's token limit
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(transcript)

    # MAP — summarize each chunk with the map prompt
    map_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that summarizes text concisely."),
            ("human", MAP_PROMPT_TEXT),
        ])
        | llm
        | parser
    )
    chunk_summaries = [map_chain.invoke({"chunk": c}) for c in chunks]

    # REDUCE — combine all summaries into one final output
    reduce_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that creates clear, structured summaries."),
            ("human", REDUCE_PROMPT_TEXT),
        ])
        | llm
        | parser
    )
    return reduce_chain.invoke({"summaries": "\n\n---\n\n".join(chunk_summaries)})
