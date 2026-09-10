# PromptEval AI: NLP-Based Prompt Engineering Assessment System

An intelligent Natural Language Processing (NLP) system and web application designed to assess students' prompt engineering skills by comparing their generated prompts against expert-designed **Golden Prompts** using **PyTorch**, **Hugging Face Sentence-Transformers**, and multi-criteria linguistic analysis.

---

## 🧠 System Architecture & NLP Methodology

The system quantifies the semantic and structural gap between a student's candidate prompt and an expert golden prompt across four key dimensions:

1. **Dense Semantic Similarity (Sentence-Transformers & PyTorch)**:
   - Uses `sentence-transformers/all-MiniLM-L6-v2` to map prompts into dense 384-dimensional vector embeddings.
   - Computes normalized Cosine Similarity:
     $$\text{Cosine Similarity} = \frac{\mathbf{v}_{\text{student}} \cdot \mathbf{v}_{\text{golden}}}{\|\mathbf{v}_{\text{student}}\| \|\mathbf{v}_{\text{golden}}\|}$$
2. **Lexical & N-Gram Alignment**:
   - Computes TF-IDF bi-gram overlap and domain keyword coverage.
3. **Four Assessment Pillars (Weighted Rubric Engine)**:
   - **Structure (25%)**: Persona conditioning (`"Act as a..."`), delimiters (`###`, ```` ``` ````, `"""`), and sequential step workflows.
   - **Clarity (25%)**: Imperative action verbs, elimination of conversational fluff and ambiguous filler phrases.
   - **Context (25%)**: Domain criteria grounding, target audience framing, and few-shot exemplars.
   - **Completeness (25%)**: Machine-parsable output schema contracts (JSON, Markdown), negative constraints (*"what NOT to do"*), and edge-case guidance.
4. **Automated Gap Analysis & Prescriptive Remediation**:
   - Diagnoses exact missing guardrails, schemas, or role anchors.
   - Generates actionable pedagogical feedback and an optimized prompt rewrite.
5. **Output Simulation Sandbox**:
   - Compares the simulated LLM response produced by a novice prompt vs. the expert golden prompt.

---

## 📂 Project Structure

```
d:\SoumyaD\
├── .venv/               # Isolated Python virtual environment with PyTorch & HuggingFace
├── nlp_engine.py        # Core NLP assessment model engine (PyTorch + SentenceTransformers)
├── app.py               # Flask REST API server and web app host (http://localhost:5000)
├── index.html           # Modern interactive web UI connected to the NLP backend
├── requirements.txt     # Locked Python package dependencies
└── README.md            # System documentation and usage guide
```

---

## 🚀 How to Run the Project

### Option 1: Full Python Web Application (Recommended)
Activate the virtual environment and start the Flask backend:
```powershell
& "d:\SoumyaD\.venv\Scripts\python.exe" d:\SoumyaD\app.py
```
This starts the PyTorch model server and automatically opens your browser at:
```
http://localhost:5000
```
*(The status indicator at the top of the interface will confirm: **● PyTorch Sentence-Transformers Active**)*.

---

### Option 2: Command-Line (CLI) Assessment
You can also run the NLP model directly from your terminal to evaluate prompts without opening the browser:
```powershell
& "d:\SoumyaD\.venv\Scripts\python.exe" d:\SoumyaD\nlp_engine.py --scenario code --prompt "Act as a Senior Python engineer. Refactor the code for O(n) performance, add type hints and docstrings."
```

**Supported Scenarios:**
- `--scenario code`: Python Code Refactoring & Complexity
- `--scenario tutor`: Socratic AI Math Tutor
- `--scenario data`: JSON Ticket Entity Extraction
- `--scenario creative`: C-Suite Executive Briefing

---

## 📡 REST API Reference

### `POST /api/assess`
Evaluates a candidate prompt against the scenario's golden prompt.

**Request Body (`application/json`):**
```json
{
  "scenario": "code",
  "student_prompt": "Refactor this python code for better performance."
}
```

**Response (`application/json`):**
```json
{
  "status": "success",
  "source": "pytorch_sentence_transformer",
  "results": {
    "overall_score": 63,
    "performance_tier": "Intermediate",
    "semantic_similarity": 0.6214,
    "dimensions": {
      "structure": { "score": 60, "description": "..." },
      "clarity": { "score": 100, "description": "..." },
      "context": { "score": 75, "description": "..." },
      "completeness": { "score": 20, "description": "..." }
    },
    "gap_analysis": [...],
    "suggestions": [...]
  }
}
```
