# 📋 Project 3 — Meeting Notes Summariser

Paste a messy meeting transcript → get clean, structured notes with decisions, action items, and open questions — automatically.

---

## 🧠 What You Learn

| Concept | What it means |
|---------|--------------|
| **Pydantic Structured Output** | Getting a typed Python object back from an LLM instead of a string |
| **LLM Function Calling** | How OpenAI's function-calling feature guarantees structured output |
| **Schema-Driven Prompting** | Defining what you want via a data model, not just words |
| **ChatPromptTemplate** | Separating system instructions from user input |

### Why Structured Output?
Asking an LLM to "list action items" gives you text — which you have to parse, and it's different every time.

**Pydantic structured output gives you a Python object:**
```python
result = chain.invoke({"transcript": text})

result.overview          # → "The team discussed Q3 launch timeline..."
result.action_items      # → ["John will send invite today", "Emily delivers mockups by Thursday"]
result.key_decisions     # → ["Go enterprise first, expand SMB in Q4"]
```

No parsing. No regex. Just clean typed data. This is how production AI apps work.

---

## 🗂️ Files

```
03-meeting-notes-summariser/
├── app.py              ← all the code (well-commented)
└── requirements.txt    ← Python packages to install
```

---

## ⚙️ Setup & Run

```bash
# 1. Navigate to this folder
cd 03-meeting-notes-summariser

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
2. Paste your meeting transcript — or click **Load Sample** to try it instantly
3. Click **Extract Meeting Notes**
4. See your structured notes ✅

**Works with:**
- Zoom / Google Meet transcript exports
- Manually typed notes from a call
- Any conversation-style text

---

## 🔑 Key Code Concepts

```python
# 1. Define the output schema as a Pydantic model
class MeetingSummary(BaseModel):
    overview: str
    key_decisions: List[str]
    action_items: List[str]
    open_questions: List[str]
    next_meeting_topics: List[str]

# 2. Tell the LLM to return data matching that schema
structured_llm = llm.with_structured_output(MeetingSummary)

# 3. Chain and invoke — you get back a MeetingSummary object
chain = prompt | structured_llm
result = chain.invoke({"transcript": transcript})

# 4. Access typed fields directly
print(result.action_items)   # → ['John sends invite', 'Emily delivers mockups']
```
