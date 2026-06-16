import streamlit as st

from core import summarise_meeting

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Meeting Summariser", page_icon="📝")
st.title("📝 Meeting Notes Summariser")
st.caption("Paste your raw meeting transcript and get structured notes instantly.")

# ── Sidebar — API key ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    st.markdown("Get your key at [platform.openai.com](https://platform.openai.com/api-keys)")

# ── Main UI ──────────────────────────────────────────────────────────────────
transcript = st.text_area(
    "Paste meeting transcript",
    height=300,
    placeholder="Alice: Let's align on the Q3 roadmap...\nBob: I think we should prioritise the API work first...",
)

if st.button("📋 Generate Meeting Notes", use_container_width=True):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    if not transcript.strip():
        st.error("Please paste a meeting transcript.")
        st.stop()

    with st.spinner("Analysing transcript..."):
        try:
            summary = summarise_meeting(transcript, api_key)
        except Exception as e:
            st.error(f"Could not generate notes: {e}")
            st.stop()

    st.success("Notes generated!")
    st.markdown("---")

    # Display each section only if it has content
    st.subheader("📌 Overview")
    st.write(summary.overview)

    if summary.key_decisions:
        st.subheader("✅ Key Decisions")
        for item in summary.key_decisions:
            st.markdown(f"- {item}")

    if summary.action_items:
        st.subheader("📋 Action Items")
        for item in summary.action_items:
            st.markdown(f"- {item}")

    if summary.open_questions:
        st.subheader("❓ Open Questions")
        for item in summary.open_questions:
            st.markdown(f"- {item}")

    if summary.next_meeting_topics:
        st.subheader("🔜 Next Meeting Topics")
        for item in summary.next_meeting_topics:
            st.markdown(f"- {item}")
