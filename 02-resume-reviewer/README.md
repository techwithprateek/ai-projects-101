# 📄 Project 2 — AI Resume Reviewer

Upload your resume as a PDF → the AI tells you what's strong, what's weak, and exactly what to fix.

---

## 🧠 What You Learn

| Concept | What it means |
|---------|--------------|
| **Structured Prompt Engineering** | How to force an LLM to always return the same format |
| **PDF Text Extraction** | Reading text out of a PDF file with Python |
| **LCEL (LangChain Expression Language)** | The modern `prompt \| llm` pipe syntax |
| **PromptTemplate** | Reusable prompts with placeholders you fill in at runtime |

### Why Structured Prompts?
If you ask "review my resume", the LLM gives different structure every time — hard to read, hard to act on.

**The fix:** put the exact format you want *inside* the prompt itself:
```
Provide your review in EXACTLY this format:
## ✅ Strengths
...
## ⚠️ Weaknesses
...
```
Now the output is consistent every single time. This is one of the most useful prompt engineering skills.

---

## 🗂️ Files

```
02-resume-reviewer/
├── app.py              ← all the code (well-commented)
└── requirements.txt    ← Python packages to install
```

---

## ⚙️ Setup & Run

```bash
# 1. Navigate to this folder
cd 02-resume-reviewer

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

1. Enter your OpenAI API key in the **sidebar**
2. (Optional) Type the job role you're targeting — makes feedback much more specific
3. Upload your resume as a **PDF**
4. Click **Review My Resume**
5. Read the structured feedback ✅

> **Important:** Your PDF must contain selectable text. If you can't highlight text in the PDF, it's a scanned image and text extraction won't work.

---

## 🔑 Key Code Concepts

```python
# 1. Read PDF text
reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
text = "".join(page.extract_text() for page in reader.pages)

# 2. Define a reusable prompt with placeholders
prompt = PromptTemplate(
    input_variables=["resume", "job_context"],
    template="Review this resume:\n{resume}\n..."
)

# 3. LCEL pipe syntax — cleaner than old LLMChain
chain = prompt | llm
result = chain.invoke({"resume": text, "job_context": "..."})
```
