import io

import PyPDF2
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Extract plain text from an uploaded PDF file
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> str:
    """
    PyPDF2 reads the PDF page by page and extracts the raw text.

    uploaded_file is a Streamlit UploadedFile object — we wrap it
    in io.BytesIO so PyPDF2 can read it like a file on disk.
    """
    # Reset file pointer to the start (important if file was read before)
    uploaded_file.seek(0)

    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))

    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # some pages may be images — skip if no text
            full_text += page_text + "\n"

    return full_text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC: Review the resume with a structured prompt
# ─────────────────────────────────────────────────────────────────────────────
def review_resume(resume_text: str, api_key: str, job_role: str = "") -> str:
    """
    Structured Prompt Engineering — the key to getting consistent output.

    The Problem:
        If you just ask "review my resume", the LLM gives a different
        structure every time. Hard to read, hard to act on.

    The Solution:
        Define EXACTLY what sections you want in the prompt itself.
        Use markdown headers so the output is always formatted the same way.
        This is called "output formatting via prompt" — a core skill.

    LCEL Syntax (LangChain Expression Language):
        chain = prompt | llm
        This "pipes" the formatted prompt directly into the LLM.
        It's cleaner than the older LLMChain approach.
    """
    llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.3)

    # Build optional context if the user gave a job role
    job_context = (
        f"The candidate is targeting this role: **{job_role}**\n"
        f"Tailor your feedback specifically for this role.\n\n"
        if job_role
        else ""
    )

    # ── The Prompt Template ───────────────────────────────────────────────────
    # {resume} and {job_context} are placeholders — LangChain fills them in.
    # The exact formatting in the template is intentional — we're telling
    # the model the output format we want via example structure.
    prompt = PromptTemplate(
        input_variables=["resume", "job_context"],
        template="""You are a professional resume reviewer with 10+ years of hiring experience.
Review the resume below and provide honest, specific, actionable feedback.

{job_context}RESUME:
{resume}

---

Provide your review in EXACTLY this format:

## ✅ Strengths
List 3 things this resume does well.
- 
- 
- 

## ⚠️ Weaknesses
List 3 things that are missing or need improvement.
- 
- 
- 

## 💡 Actionable Suggestions
List 5 specific changes the candidate should make.
1. 
2. 
3. 
4. 
5. 

## 🎯 Overall Score: X/10
One sentence explaining the score.

## 📌 The Single Most Important Fix
One sentence — if they could only do one thing, what should it be?""",
    )

    # LCEL: pipe prompt into llm, then parse the output to a plain string
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke(
        {
            "resume": resume_text[:6000],  # trim very long resumes to avoid token limits
            "job_context": job_context,
        }
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Resume Reviewer", page_icon="📄", layout="centered")

st.title("📄 AI Resume Reviewer")
st.markdown(
    "Upload your resume and get **instant AI feedback** on strengths, weaknesses, and exactly what to fix."
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
    st.markdown("**💡 Tips**")
    st.markdown(
        """
    - Your PDF must contain selectable text (not a scanned image)
    - Adding a target job role makes the feedback much more useful
    - The AI focuses on structure, content, and clarity — not design
    """
    )

# ── Main: file uploader + job role ───────────────────────────────────────────
job_role = st.text_input(
    "Target Job Role (optional but recommended)",
    placeholder="e.g.  Data Analyst,  Software Engineer,  Product Manager",
)

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf"],
    help="Only PDF files are supported.",
)

# Show a preview of the file if uploaded
if uploaded_file:
    st.success(f"✅ File uploaded: **{uploaded_file.name}**")

if st.button("🔍 Review My Resume", type="primary", use_container_width=True):

    # Validation
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        st.stop()
    if not uploaded_file:
        st.error("⚠️ Please upload a PDF resume.")
        st.stop()

    # Step 1: Extract text from PDF
    with st.spinner("📖 Reading your resume..."):
        try:
            resume_text = extract_text_from_pdf(uploaded_file)
        except Exception as e:
            st.error(f"❌ Failed to read PDF: {e}")
            st.stop()

    if not resume_text:
        st.error(
            "❌ Couldn't extract text from this PDF. "
            "Make sure it's not a scanned image — try copy-pasting text from it first."
        )
        st.stop()

    word_count = len(resume_text.split())
    st.info(f"📄 Resume loaded — **{word_count} words** extracted. Analysing...")

    # Step 2: Get AI review
    with st.spinner("🤖 AI is reviewing your resume..."):
        try:
            feedback = review_resume(resume_text, api_key, job_role)
        except Exception as e:
            st.error(f"❌ Review failed: {e}")
            st.stop()

    # Display results
    st.success("✅ Review complete!")
    st.markdown("---")
    st.markdown(feedback)

    # Show extracted text so user can verify extraction worked correctly
    with st.expander("📜 View Extracted Resume Text"):
        st.caption("This is what the AI saw — verify it looks correct")
        st.text(resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""))
