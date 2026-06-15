# AI Projects 101 🤖

Three beginner-friendly AI projects you can build, run, and add to your resume — each one teaches you a core concept in building real LLM-powered apps.

---

## 📦 Projects

| # | Project | What it Does | Key Concept |
|---|---------|-------------|-------------|
| 1 | [YouTube Summariser](./01-youtube-summariser/) | Paste a YouTube link → get a clean summary | Map-Reduce summarization for long text |
| 2 | [Resume Reviewer](./02-resume-reviewer/) | Upload your resume → AI reviews it | Structured prompt engineering |
| 3 | [Meeting Notes Summariser](./03-meeting-notes-summariser/) | Paste a transcript → get structured notes | Pydantic structured output |

---

## 🧰 Prerequisites

Before running any project, make sure you have:

1. **Python 3.9+** installed → [Download](https://www.python.org/downloads/)
   ```bash
   python --version   # should say 3.9 or higher
   ```

2. **An OpenAI API key** → [Get one here](https://platform.openai.com/api-keys)
   - Create an account → go to API keys → create a new key
   - You'll be billed per use (each project run costs < $0.05)

3. **pip** (comes with Python — used to install packages)

---

## 🚀 How to Run Any Project

Each project is self-contained. Pick one and follow these steps:

```bash
# 1. Clone this repo (if you haven't already)
git clone https://github.com/techwithprateek/ai-projects-101.git
cd ai-projects-101

# 2. Go into the project folder
cd 01-youtube-summariser    # or 02-resume-reviewer, or 03-meeting-notes-summariser

# 3. (Recommended) Create a virtual environment so packages don't conflict
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.  
Enter your OpenAI API key in the sidebar and you're good to go.

---

## 🗂️ Repo Structure

```
ai-projects-101/
├── 01-youtube-summariser/
│   ├── app.py              ← main application code
│   ├── requirements.txt    ← Python packages needed
│   └── README.md           ← project-specific guide
│
├── 02-resume-reviewer/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
└── 03-meeting-notes-summariser/
    ├── app.py
    ├── requirements.txt
    └── README.md
```

---

## 🧠 What You'll Learn Across All 3 Projects

| Concept | Where |
|---------|-------|
| Calling OpenAI via LangChain | All projects |
| Map-Reduce summarization (handling long text) | Project 1 |
| Extracting text from external sources (YouTube, PDF) | Projects 1 & 2 |
| Structured prompt engineering | Project 2 |
| Pydantic structured output from LLMs | Project 3 |
| Building UIs with Streamlit | All projects |

---

## ❓ Common Issues

**"ModuleNotFoundError"** → Make sure you ran `pip install -r requirements.txt` inside the right folder.

**"AuthenticationError"** → Your OpenAI API key is wrong or has no credits. Check at [platform.openai.com](https://platform.openai.com).

**"TranscriptsDisabled"** → The YouTube video has disabled captions. Try a different video.

**Streamlit doesn't open** → Go to `http://localhost:8501` manually in your browser.
