import streamlit as st

from core import extract_text_from_pdf, review_resume

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Resume Reviewer", page_icon="📄")
st.title("📄 AI Resume Reviewer")
st.caption("Upload your resume (PDF) and get instant, actionable feedback.")

# ── Sidebar — API key ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    st.markdown("Get your key at [platform.openai.com](https://platform.openai.com/api-keys)")

# ── Main UI ──────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_role      = st.text_input(
    "Target job role (optional)",
    placeholder="e.g. Data Engineer at a startup",
)

if st.button("🔍 Review My Resume", use_container_width=True):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    if not uploaded_file:
        st.error("Please upload a PDF resume.")
        st.stop()

    with st.spinner("Reading your resume..."):
        try:
            resume_text = extract_text_from_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Could not read PDF: {e}")
            st.stop()

    if not resume_text.strip():
        st.error("The PDF appears to be empty or is image-based (scanned). Please upload a text-based PDF.")
        st.stop()

    with st.spinner("Reviewing with AI..."):
        try:
            review = review_resume(resume_text, api_key, job_role)
        except Exception as e:
            st.error(f"Review failed: {e}")
            st.stop()

    st.success("Review complete!")
    st.markdown("---")
    st.markdown(review)

    # Let the user download the review as a text file
    st.download_button(
        "⬇️ Download Review",
        data=review,
        file_name="resume_review.md",
        mime="text/markdown",
    )
