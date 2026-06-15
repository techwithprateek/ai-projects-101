from typing import List

import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SCHEMA: Define exactly what we want the LLM to return
# ─────────────────────────────────────────────────────────────────────────────
class MeetingSummary(BaseModel):
    """
    Pydantic Structured Output — the cleanest way to get structured data from an LLM.

    The Problem:
        If you ask an LLM "give me action items", it might return them as
        bullet points, numbered lists, paragraphs — different every time.
        Parsing that string is messy and fragile.

    The Solution:
        Define a Pydantic model (like a typed schema) and tell LangChain:
        "I want the output to match THIS structure."

        LangChain uses function-calling under the hood to force the LLM
        to respond with JSON that matches the model. It then automatically
        parses that JSON into a Python object.

        Result: you get a MeetingSummary object with .overview, .action_items,
        etc. — proper typed Python, not a string you have to parse yourself.
    """

    overview: str = Field(
        description="A 2-3 sentence summary of what the meeting was about and the main outcome"
    )
    key_decisions: List[str] = Field(
        description="Important decisions that were agreed on during the meeting"
    )
    action_items: List[str] = Field(
        description="Tasks assigned to people — include the owner and deadline if mentioned "
        "(e.g. 'John will send the proposal by Friday')"
    )
    open_questions: List[str] = Field(
        description="Topics or questions that were raised but not resolved"
    )
    next_meeting_topics: List[str] = Field(
        description="Topics explicitly deferred to the next meeting, if any"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC: Extract structured notes from a messy transcript
# ─────────────────────────────────────────────────────────────────────────────
def summarise_meeting(transcript: str, api_key: str) -> MeetingSummary:
    """
    How structured output works in LangChain:

        1. Create a normal ChatOpenAI LLM
        2. Call .with_structured_output(MeetingSummary)
           → This tells LangChain to use OpenAI's function-calling feature
             to guarantee the output matches our Pydantic schema
        3. Pipe a prompt into it — the output is a MeetingSummary object,
           not a string

    This is the modern, reliable way to extract structured info from LLMs.
    """
    llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)

    # .with_structured_output() wraps the LLM so it always returns a MeetingSummary
    structured_llm = llm.with_structured_output(MeetingSummary)

    # ChatPromptTemplate lets you define system + human messages separately
    # "system" = instructions for the LLM  |  "human" = the actual input
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at analysing meeting transcripts.
Extract the key information from the transcript provided.
Be concise and specific. Use short, clear bullet-point style for lists.
If a field has no relevant content, return an empty list.""",
            ),
            ("human", "Here is the meeting transcript:\n\n{transcript}"),
        ]
    )

    # Chain: format prompt → send to structured LLM → returns MeetingSummary object
    chain = prompt | structured_llm

    return chain.invoke({"transcript": transcript})


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE TRANSCRIPT — lets users try the app without having a real transcript
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_TRANSCRIPT = """
John: Okay let's get started. We need to talk about the Q3 product launch timeline.

Sarah: Right. I think we're behind on the design mockups. Emily was supposed to finish those last week.

Emily: Sorry about that — the design review got pushed. I'll have them done by Thursday for sure.

John: Okay, that's decided. Emily delivers mockups by Thursday. What about the marketing copy?

Mark: I'm about 80% done. Should be ready by end of week. One question though — are we targeting enterprise or SMB customers for this launch?

John: Good question. Let's go enterprise first and expand to SMB in Q4. That's the decision.

Sarah: What about the pricing page? We haven't aligned on that yet.

John: That's a good point — we don't have time today. Can we make that the main topic for next week?

Mark: Works for me. We should also loop in the sales team before deciding on enterprise pricing.

John: Agreed. Mark, can you set that up?

Mark: Sure, I'll send the invite today.

John: Great. So to recap — Emily has mockups by Thursday, Mark finishes copy by Friday and sends a sales team invite today. Pricing page discussion moves to next week's meeting.
"""


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Notes Summariser", page_icon="📋", layout="centered"
)

st.title("📋 Meeting Notes Summariser")
st.markdown(
    "Paste any messy meeting transcript and get **clean, structured notes** — "
    "decisions, action items, and open questions, all extracted automatically."
)

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
    st.markdown("**💡 Works great with:**")
    st.markdown(
        """
    - Zoom / Google Meet transcript exports
    - Manually typed notes from a call
    - Messy Slack threads summarising a meeting
    - Any conversation-style text
    """
    )

# ── Main: transcript input ───────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("Paste your meeting transcript below")
with col2:
    load_sample = st.button("Load Sample")

# Load sample transcript if button clicked
default_text = SAMPLE_TRANSCRIPT if load_sample else ""

transcript = st.text_area(
    "Meeting transcript",
    value=default_text,
    height=280,
    placeholder="Paste your meeting transcript here...\n\nExample:\nJohn: Let's align on the launch date.\nSarah: I think mid-October works...",
    label_visibility="collapsed",
)

if st.button("📋 Extract Meeting Notes", type="primary", use_container_width=True):

    # Validation
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        st.stop()
    if not transcript.strip():
        st.error("⚠️ Please paste a meeting transcript (or click 'Load Sample' to try it out).")
        st.stop()

    # Extract structured notes
    with st.spinner("🤖 Analysing your meeting transcript..."):
        try:
            result = summarise_meeting(transcript, api_key)
        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
            st.stop()

    st.success("✅ Done! Here are your structured meeting notes:")
    st.markdown("---")

    # ── Overview (full width) ─────────────────────────────────────────────────
    st.markdown("### 📌 Meeting Overview")
    st.info(result.overview)

    # ── Two-column layout for the lists ──────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("### ✅ Key Decisions")
        if result.key_decisions:
            for item in result.key_decisions:
                st.markdown(f"- {item}")
        else:
            st.caption("No key decisions found")

        st.markdown("### ❓ Open Questions")
        if result.open_questions:
            for item in result.open_questions:
                st.markdown(f"- {item}")
        else:
            st.caption("No open questions found")

    with right:
        st.markdown("### 📋 Action Items")
        if result.action_items:
            for item in result.action_items:
                st.markdown(f"- {item}")
        else:
            st.caption("No action items found")

        st.markdown("### 🔜 Next Meeting Topics")
        if result.next_meeting_topics:
            for item in result.next_meeting_topics:
                st.markdown(f"- {item}")
        else:
            st.caption("None mentioned")
