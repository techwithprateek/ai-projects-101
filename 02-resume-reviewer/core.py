import io
from pathlib import Path

import PyPDF2
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from config import MAX_RESUME_CHARS, MODEL, TEMPERATURE

# Load the review prompt template from the prompts/ folder at import time
_PROMPTS = Path(__file__).parent / "prompts"
REVIEW_PROMPT_TEXT = (_PROMPTS / "review.txt").read_text()


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract plain text from a Streamlit-uploaded PDF file.
    Returns all pages joined as a single string.
    """
    uploaded_file.seek(0)
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    pages  = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def review_resume(resume_text: str, api_key: str, job_role: str = "") -> str:
    """
    Send the resume to the LLM with a structured prompt and return the review.
    If a job role is provided, the prompt includes it for more targeted feedback.
    """
    llm = ChatOpenAI(api_key=api_key, model=MODEL, temperature=TEMPERATURE)

    job_context = (
        f"The candidate is targeting this role: **{job_role}**\n"
        f"Tailor your feedback specifically for this role.\n\n"
        if job_role
        else ""
    )

    chain = (
        PromptTemplate(
            input_variables=["resume", "job_context"],
            template=REVIEW_PROMPT_TEXT,
        )
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "resume":      resume_text[:MAX_RESUME_CHARS],
        "job_context": job_context,
    })
