from pathlib import Path
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import MODEL, TEMPERATURE

# Load the extraction prompt from the prompts/ folder at import time
_PROMPTS = Path(__file__).parent / "prompts"
EXTRACT_PROMPT_TEXT = (_PROMPTS / "extract.txt").read_text()


class MeetingSummary(BaseModel):
    """
    Defines the exact structure we want back from the LLM.
    LangChain uses this schema to force the model to respond
    with typed, structured data instead of free-form text.
    """
    overview:            str        = Field(description="2-3 sentence summary of the meeting and its main outcome")
    key_decisions:       List[str]  = Field(description="Decisions that were agreed on during the meeting")
    action_items:        List[str]  = Field(description="Tasks assigned to people — include owner and deadline if mentioned")
    open_questions:      List[str]  = Field(description="Topics or questions raised but not resolved")
    next_meeting_topics: List[str]  = Field(description="Topics explicitly deferred to the next meeting")


def summarise_meeting(transcript: str, api_key: str) -> MeetingSummary:
    """
    Extract structured meeting notes from a transcript.
    Returns a MeetingSummary object with typed fields — not a raw string.
    """
    import warnings

    llm = ChatOpenAI(api_key=api_key, model=MODEL, temperature=TEMPERATURE)

    # with_structured_output() tells LangChain to use OpenAI function-calling
    # to guarantee the response matches our MeetingSummary Pydantic schema
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structured_llm = llm.with_structured_output(MeetingSummary)

    chain = (
        ChatPromptTemplate.from_messages([
            ("system", EXTRACT_PROMPT_TEXT),
            ("human",  "Here is the meeting transcript:\n\n{transcript}"),
        ])
        | structured_llm
    )

    return chain.invoke({"transcript": transcript})
