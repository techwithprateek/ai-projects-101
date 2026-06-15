# 🎬 Project 1 — YouTube Video Summariser

Paste any YouTube link → get a clean, structured summary of the entire video without watching it.

---

## 🧠 What You Learn

| Concept | What it means |
|---------|--------------|
| **Map-Reduce Summarization** | How to summarize text that's too long for one LLM call |
| **Text Splitting** | Chunking long documents into overlapping pieces |
| **LangChain Chains** | Chaining multiple LLM calls together automatically |
| **External Data Sources** | Fetching real data (transcripts) from the internet |

### Why Map-Reduce?
LLMs have a token limit (~4,000 tokens for GPT-3.5). A 1-hour YouTube video can be 15,000+ words — way too long to send in one request.

**Map-Reduce solves this:**
1. **MAP:** Split the transcript → summarize each chunk separately
2. **REDUCE:** Combine all chunk summaries → one final summary

This is the same pattern used in big data processing (Hadoop, Spark).

---

## 🗂️ Files

```
01-youtube-summariser/
├── app.py              ← all the code (well-commented)
└── requirements.txt    ← Python packages to install
```

---

## ⚙️ Setup & Run

```bash
# 1. Navigate to this folder
cd 01-youtube-summariser

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# 3. Install packages
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Your browser will open at `http://localhost:8501`.

---

## 🎮 How to Use

1. Enter your OpenAI API key in the **sidebar** (left side)
2. Paste a YouTube URL in the text box
3. Click **Summarise Video**
4. Wait ~20-30 seconds for longer videos
5. Read your summary ✅

---

## 💡 Try These Videos

- Any tutorial or lecture on YouTube
- TED Talks
- Podcast recordings with captions
- Conference talks

> **Note:** The video must have captions (auto-generated is fine). If you get a `TranscriptsDisabled` error, the creator has turned off captions — try a different video.

---

## 🔑 Key Code Concepts

```python
# 1. Split long text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

# 2. Map-Reduce chain — LangChain handles the loop automatically
chain = load_summarize_chain(llm, chain_type="map_reduce", ...)

# 3. Run it
summary = chain.run(docs)
```
