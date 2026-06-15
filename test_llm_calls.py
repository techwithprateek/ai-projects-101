"""
Run this script yourself to test all 3 projects end-to-end with real LLM calls.

Usage:
    cd /Users/Prateek/Development/GenAI/ai-projects-101
    source venv/bin/activate
    OPENAI_API_KEY=sk-... python3 test_llm_calls.py
"""
import os, sys

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("❌ No API key found.")
    print("   Run with: OPENAI_API_KEY=sk-... python3 test_llm_calls.py")
    sys.exit(1)

print(f"✅ API key loaded (ends in ...{api_key[-4:]})\n")

# ── PROJECT 1: YouTube Summariser ────────────────────────────────────────────
print("=" * 60)
print("PROJECT 1 — YouTube Summariser (map-reduce)")
print("=" * 60)
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Short public video with captions: "Me at the zoo" (19 seconds, ~40 words)
VIDEO_ID = "jNQXAC9IVRw"
print(f"  Video: https://youtu.be/{VIDEO_ID}")

api = YouTubeTranscriptApi()
snippets = api.fetch(VIDEO_ID)
transcript = " ".join(s.text for s in snippets)
print(f"  Transcript: {len(transcript.split())} words — '{transcript[:80].strip()}...'")

llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)
parser = StrOutputParser()

splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
chunks = splitter.split_text(transcript)

map_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes text concisely."),
    ("human", "Summarise this transcript section:\n\n{chunk}\n\nSUMMARY:"),
])
map_chain = map_prompt | llm | parser
chunk_summaries = [map_chain.invoke({"chunk": c}) for c in chunks]

reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that creates clear summaries."),
    ("human", """Combine these into a final summary with Overview, Key Points, Main Takeaway:

{summaries}

FINAL SUMMARY:"""),
])
reduce_chain = reduce_prompt | llm | parser
summary = reduce_chain.invoke({"summaries": "\n\n---\n\n".join(chunk_summaries)})

print(f"\n  ✅ SUMMARY OUTPUT:\n")
print(summary)

# ── PROJECT 2: Resume Reviewer ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("PROJECT 2 — Resume Reviewer (structured prompt)")
print("=" * 60)
from langchain_core.prompts import PromptTemplate

SAMPLE_RESUME = """
John Smith | john@email.com | linkedin.com/in/johnsmith

EXPERIENCE
Data Analyst — Acme Corp (2021–Present)
- Built dashboards in Tableau for sales team
- Wrote SQL queries for reporting
- Analyzed customer data

Junior Analyst — StartupXYZ (2019–2021)
- Excel data analysis
- Created weekly reports

EDUCATION
B.Sc. Computer Science — State University (2019)

SKILLS
Python, SQL, Excel, Tableau
"""

prompt = PromptTemplate(
    input_variables=["resume", "job_context"],
    template="""You are a professional resume reviewer. Review this resume.

{job_context}RESUME:
{resume}

Provide feedback in this EXACT format:

## ✅ Strengths
- 
- 
- 

## ⚠️ Weaknesses
- 
- 
- 

## 💡 Actionable Suggestions
1. 
2. 
3. 
4. 
5. 

## 🎯 Overall Score: X/10
(one sentence)

## 📌 The Single Most Important Fix
(one sentence)""",
)
chain = prompt | llm | parser
feedback = chain.invoke({
    "resume": SAMPLE_RESUME,
    "job_context": "Targeting role: Senior Data Analyst\n\n",
})
print(f"\n  ✅ REVIEW OUTPUT:\n")
print(feedback)

# ── PROJECT 3: Meeting Notes Summariser ─────────────────────────────────────
print("\n" + "=" * 60)
print("PROJECT 3 — Meeting Notes (Pydantic structured output)")
print("=" * 60)
from pydantic import BaseModel, Field
from typing import List
import warnings

class MeetingSummary(BaseModel):
    overview: str = Field(description="2-3 sentence summary of the meeting")
    key_decisions: List[str] = Field(description="Decisions made")
    action_items: List[str] = Field(description="Tasks assigned with owner/deadline if mentioned")
    open_questions: List[str] = Field(description="Unresolved questions")
    next_meeting_topics: List[str] = Field(description="Topics deferred to next meeting")

SAMPLE_TRANSCRIPT = """
John: Let's get started. We need to align on the Q3 product launch timeline.
Sarah: I think we're behind on design mockups. Emily was supposed to finish those last week.
Emily: Sorry — design review was delayed. I'll have them done by Thursday for sure.
John: Decided — Emily delivers mockups by Thursday. What about marketing copy?
Mark: 80% done, ready by end of week. Quick question — enterprise or SMB for this launch?
John: Enterprise first, SMB in Q4. That's the decision.
Sarah: What about the pricing page? We haven't aligned on that yet.
John: No time today. Let's make that the main topic for next week.
Mark: We should loop in the sales team for enterprise pricing too.
John: Agreed. Mark, can you set that up?
Mark: Sure, I'll send the invite today.
John: Great. Recap — Emily: mockups by Thursday. Mark: copy by Friday, sales invite today. Pricing page next week.
"""

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    structured_llm = llm.with_structured_output(MeetingSummary)

from langchain_core.prompts import ChatPromptTemplate
meeting_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at analysing meeting transcripts. Extract key information. Be concise."),
    ("human", "Meeting transcript:\n\n{transcript}"),
])
meeting_chain = meeting_prompt | structured_llm
result = meeting_chain.invoke({"transcript": SAMPLE_TRANSCRIPT})

print(f"\n  ✅ STRUCTURED OUTPUT:\n")
print(f"  Overview:            {result.overview}")
print(f"  Key Decisions:       {result.key_decisions}")
print(f"  Action Items:        {result.action_items}")
print(f"  Open Questions:      {result.open_questions}")
print(f"  Next Meeting Topics: {result.next_meeting_topics}")

# Verify it's a real MeetingSummary object with all fields
assert isinstance(result, MeetingSummary)
assert isinstance(result.key_decisions, list)
assert isinstance(result.action_items, list)
print("\n  ✅ All fields are correctly typed Python objects (not strings)")

print("\n" + "=" * 60)
print("  ✅ ALL 3 PROJECTS TESTED WITH REAL LLM CALLS")
print("=" * 60)
