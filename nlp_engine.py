"""
NLP-Based Prompt Engineering Assessment Engine
==============================================
Uses Hugging Face Sentence-Transformers, PyTorch, and linguistic analysis to evaluate
student-generated prompts against expert golden prompts across 4 core dimensions:
1. Structure (Persona, delimiters, step sequence)
2. Clarity (Imperative directives, low ambiguity, conciseness)
3. Context (Domain background, target audience, few-shot examples)
4. Completeness (Output schema, negative constraints, edge cases)
"""

import os
import re
import math
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Load pretrained embedding model (cached locally in Hugging Face cache)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model_instance: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy loader for SentenceTransformer to optimize memory and startup."""
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer(MODEL_NAME)
    return _model_instance


# Benchmark Scenarios with Expert Golden Prompts
BENCHMARK_SCENARIOS = {
    "code": {
        "title": "Python Code Refactoring",
        "category": "Software Engineering",
        "golden_prompt": (
            "You are a Senior Staff Python Engineer and Code Reviewer.\n"
            "Your task is to refactor the provided Python code snippet to optimize runtime complexity, enforce PEP 8 style standards, and enhance readability.\n\n"
            "Context & Rules:\n"
            "- The target environment is Python 3.12+.\n"
            "- The refactored solution must be accompanied by comprehensive docstrings in Google format with type annotations.\n"
            "- Include an asymptotic analysis (Big-O time and space complexity) comparing the original vs. refactored implementation.\n\n"
            "Constraints:\n"
            "- Do not introduce external dependencies outside the Python Standard Library.\n"
            "- If edge cases (e.g. empty lists, null inputs, negative integers) exist, handle them with clear exceptions.\n"
            "- Output Format: Provide your response in valid Markdown using code blocks for the refactored code followed by bulleted explanations."
        ),
        "outputs": {
            "student": (
                "Here is a refactored version:\n\n"
                "def calc(a):\n"
                "    return [x*2 for x in a if x > 0]\n\n"
                "I made it use a list comprehension which is faster in Python!"
            ),
            "golden": (
                "### Refactored Python Implementation\n\n"
                "```python\n"
                "from typing import Sequence\n\n"
                "def process_positive_doubles(numbers: Sequence[int | float]) -> list[int | float]:\n"
                '    """Filters positive values from an input sequence and doubles each element.\n\n'
                "    Args:\n"
                "        numbers: A sequence of numeric elements (ints or floats).\n\n"
                "    Returns:\n"
                "        A list containing doubled values of all strictly positive inputs.\n\n"
                "    Raises:\n"
                '        TypeError: If the input sequence is None or contains non-numeric elements.\n'
                '    """\n'
                "    if numbers is None:\n"
                '        raise TypeError("Input sequence cannot be None")\n\n'
                "    return [num * 2 for num in numbers if num > 0]\n"
                "```\n\n"
                "### Asymptotic Analysis\n"
                "- **Original Complexity**: O(n²) time / O(n) space\n"
                "- **Refactored Complexity**: O(n) time / O(n) space\n\n"
                "### Key Enhancements\n"
                "1. Enforced strict PEP 484 type annotations (`Sequence[int | float]`).\n"
                "2. Handled edge case where `numbers is None`.\n"
                "3. Standardized Google-style docstrings with explicit contracts."
            ),
        },
        "improved_prompt": (
            "You are a Senior Staff Python Engineer.\n"
            "Refactor the provided Python code to optimize runtime complexity to O(n) and conform to PEP 8 standards.\n\n"
            "Requirements:\n"
            "- Add comprehensive type hints and Google-style docstrings.\n"
            "- Handle edge cases including empty inputs and invalid types.\n"
            "- Provide a Big-O complexity comparison.\n"
            "- Format the response in structured Markdown."
        )
    },

    "tutor": {
        "title": "Socratic AI Math Tutor",
        "category": "Pedagogy & Education",
        "golden_prompt": (
            "You are a patient, encouraging Socratic Math Tutor specializing in high school calculus.\n"
            "Your goal is to guide the student toward discovering the concept of derivatives and instantaneous rate of change on their own.\n\n"
            "Pedagogical Rules:\n"
            "- Never provide direct answers or final equations upfront.\n"
            "- Ask targeted, thought-provoking guiding questions one step at a time.\n"
            "- Connect mathematical concepts to real-world physical intuition (such as speedometer readings in an accelerating car).\n"
            "- If the student makes a misconception, validate their effort, gently highlight the discrepancy, and ask a scaffolding question.\n\n"
            "Output Style:\n"
            "- Friendly, warm, and conversational.\n"
            "- Limit each response to 2-3 sentences followed by exactly ONE reflective question."
        ),
        "outputs": {
            "student": (
                "Calculus is about change! A derivative is the slope of a curve: "
                "f'(x) = lim (f(x+h)-f(x))/h as h approaches 0. Do you get it?"
            ),
            "golden": (
                "Welcome! Imagine you're on a highway and look down at your speedometer reading 65 mph. "
                "That speedometer tells you how fast you're going at that exact split second, rather than your average speed over the entire two-hour trip.\n\n"
                "If average speed is distance divided by elapsed time, how might we calculate speed if the elapsed time shrinks down to almost zero seconds?"
            )
        },
        "improved_prompt": (
            "You are a Socratic high school calculus tutor.\n"
            "Guide the student to intuitively understand derivatives and instantaneous rate of change.\n\n"
            "Rules:\n"
            "- Never reveal formulas directly; use the analogy of a car speedometer.\n"
            "- Guide step-by-step with 1 targeted question per turn.\n"
            "- Be encouraging, concise, and pedagogical."
        )
    },

    "data": {
        "title": "JSON Data Extraction",
        "category": "Information Extraction",
        "golden_prompt": (
            "You are a specialized Data Extraction Engine.\n"
            "Analyze the unformatted customer inquiry text and extract key entities into a strictly formatted JSON object.\n\n"
            "Target Schema:\n"
            "{\n"
            '  "customer_id": string or null,\n'
            '  "sentiment": "positive" | "neutral" | "negative",\n'
            '  "urgency": "low" | "medium" | "high",\n'
            '  "product_mentions": string[],\n'
            '  "core_issue": string,\n'
            '  "action_required": boolean\n'
            "}\n\n"
            "Constraints:\n"
            "- Output ONLY valid, parsable RFC 8259 JSON.\n"
            "- Do not wrap the JSON in commentary, introductory remarks, or conversational filler.\n"
            "- If an entity is missing or ambiguous, populate with null rather than guessing."
        ),
        "outputs": {
            "student": (
                "Sure! Here is the JSON data you requested:\n"
                "{\n"
                '  "customer": "John Doe",\n'
                '  "sentiment": "bad",\n'
                '  "product": "AirFlow 3000"\n'
                "}\n"
                "Hope this helps! Let me know if you need anything else!"
            ),
            "golden": (
                "{\n"
                '  "customer_id": "CUST-84920",\n'
                '  "sentiment": "negative",\n'
                '  "urgency": "high",\n'
                '  "product_mentions": ["AirFlow 3000 Pro", "HEPA Filter B"],\n'
                '  "core_issue": "Device motor smoking upon initial power-up",\n'
                '  "action_required": true\n'
                "}"
            )
        },
        "improved_prompt": (
            "You are an automated Data Extraction Engine.\n"
            "Extract customer ticket entities into a strict RFC 8259 JSON object matching this schema:\n"
            "{ customer_id, sentiment, urgency, product_mentions: [], core_issue, action_required: boolean }\n\n"
            "Constraints:\n"
            "- Output ONLY raw valid JSON without conversational wrapper text.\n"
            "- Fill missing fields with null."
        )
    },

    "creative": {
        "title": "Executive Briefing Synthesis",
        "category": "Executive Communications",
        "golden_prompt": (
            "You are a Chief Strategy Officer and Executive Communications Director.\n"
            "Your task is to synthesize the provided 20-page quarterly performance report into a concise 1-page C-suite Executive Briefing.\n\n"
            "Structure Requirements:\n"
            "1. Executive Summary (Max 3 bullet points highlighting top strategic milestones).\n"
            "2. Key Quantitative KPI Delta (Revenue, Net Retention, CAC, Runway in a formatted Markdown table).\n"
            "3. Critical Operational Risks & Headwinds.\n"
            "4. Strategic Decisions Required from the Board.\n\n"
            "Tone & Format:\n"
            "- Direct, data-grounded, objective, and executive-ready.\n"
            "- Use bolding for figures and percentage deltas.\n"
            "- Total response length must not exceed 450 words."
        ),
        "outputs": {
            "student": (
                "Summary:\n"
                "This quarter was great! We made more money than last year and our team grew. "
                "There were some delays with shipping but overall everything is going well. "
                "The CEO should invest more in marketing."
            ),
            "golden": (
                "# Q3 Executive Briefing: Board of Directors\n\n"
                "### 1. Executive Highlights\n"
                "- **ARR Expansion**: Surpassed **$42.5M ARR (+34% YoY)**, beating consensus by **$1.8M**.\n"
                "- **Efficiency Milestone**: Reached **Cash-Flow Breakeven** 2 quarters ahead of target.\n"
                "- **Enterprise Velocity**: Closed **14 Fortune 500 accounts**; Net Retention Rate maintained at **128%**.\n\n"
                "### 2. Core KPI Matrix\n"
                "| Metric | Q3 Actual | Q2 Actual | YoY Delta |\n"
                "| :--- | :--- | :--- | :--- |\n"
                "| **Gross Margin** | **78.4%** | 76.1% | **+230 bps** |\n"
                "| **CAC Payback** | **11.2 mo** | 14.1 mo | **-2.9 mo** |\n"
                "| **Burn Multiple**| **0.4x** | 0.9x | **-0.5x** |\n\n"
                "### 3. Key Risks & Board Action Items\n"
                "- **EU Regulatory Horizon**: GDPR-AI audit requires $400K compliance allocation.\n"
                "- **Board Decision**: Approval sought for Series C expansion reserve."
            )
        },
        "improved_prompt": (
            "You are a Chief Strategy Officer.\n"
            "Transform the report into a high-impact C-Suite Executive Briefing.\n\n"
            "Sections:\n"
            "1. Top 3 Strategic Milestones\n"
            "2. KPI Matrix (ARR, Gross Margin, CAC Payback in Markdown table)\n"
            "3. Critical Headwinds & Decisions Required\n\n"
            "Tone & Style:\n"
            "- Data-grounded, bold numbers, max 400 words."
        )
    }
}


class PromptEvaluator:
    """Core NLP Prompt Evaluator using Transformers and Linguistic Analysis."""

    def __init__(self):
        self.model = get_embedding_model()

    def evaluate(self, student_prompt: str, golden_prompt: str, scenario_key: str = "code") -> Dict[str, Any]:
        student_text = (student_prompt or "").strip()
        golden_text = (golden_prompt or "").strip()

        if not student_text:
            raise ValueError("Student prompt cannot be empty.")

        # 1. Dense Semantic Similarity using Sentence-Transformers
        student_emb = self.model.encode([student_text])[0]
        golden_emb = self.model.encode([golden_text])[0]
        
        cosine_sim = float(cosine_similarity([student_emb], [golden_emb])[0][0])
        # Bound cosine similarity nicely
        cosine_sim = max(0.0, min(1.0, cosine_sim))

        # 2. N-Gram & Vocabulary Jaccard / Overlap
        tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        try:
            tfidf_mat = tfidf.fit_transform([student_text, golden_text])
            tfidf_sim = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])
        except Exception:
            tfidf_sim = 0.0

        # 3. Sub-dimension NLP scoring
        structure_score, struct_details = self._score_structure(student_text, golden_text)
        clarity_score, clarity_details = self._score_clarity(student_text, golden_text)
        context_score, context_details = self._score_context(student_text, golden_text, scenario_key)
        completeness_score, comp_details = self._score_completeness(student_text, golden_text)

        # 4. Overall Composite Quality Score
        # 40% weight on dense semantic embeddings + 15% on each of the 4 structural rubrics
        rubric_avg = (structure_score + clarity_score + context_score + completeness_score) / 4.0
        overall_score = round((cosine_sim * 100 * 0.40) + (rubric_avg * 0.60))
        overall_score = max(10, min(100, overall_score))

        # Performance Tier Classification
        if overall_score >= 85:
            tier = "Expert Master"
            tier_color = "#10b981"
        elif overall_score >= 70:
            tier = "Proficient"
            tier_color = "#6366f1"
        elif overall_score >= 50:
            tier = "Intermediate"
            tier_color = "#f59e0b"
        else:
            tier = "Needs Refinement"
            tier_color = "#f43f5e"

        # 5. Gap Analysis & Prescriptive Remediation Generation
        gap_analysis = self._generate_gap_analysis(struct_details, clarity_details, context_details, comp_details)
        suggestions = self._generate_suggestions(struct_details, clarity_details, context_details, comp_details)

        # Get scenario details
        scenario = BENCHMARK_SCENARIOS.get(scenario_key, BENCHMARK_SCENARIOS["code"])

        return {
            "overall_score": overall_score,
            "performance_tier": tier,
            "tier_color": tier_color,
            "semantic_similarity": round(cosine_sim, 4),
            "lexical_similarity": round(tfidf_sim, 4),
            "completeness_delta": f"-{max(0, 100 - overall_score)}%",
            "dimensions": {
                "structure": {
                    "score": round(structure_score),
                    "description": "System persona framing, instruction sequence, and delimiter usage.",
                    "details": struct_details
                },
                "clarity": {
                    "score": round(clarity_score),
                    "description": "Direct imperative directives, low lexical perplexity, and conciseness.",
                    "details": clarity_details
                },
                "context": {
                    "score": round(context_score),
                    "description": "Domain-specific grounding, edge cases, and background assumptions.",
                    "details": context_details
                },
                "completeness": {
                    "score": round(completeness_score),
                    "description": "Output schema contracts, negative constraints, and safety guardrails.",
                    "details": comp_details
                }
            },
            "gap_analysis": gap_analysis,
            "suggestions": suggestions,
            "improved_prompt": scenario.get("improved_prompt", ""),
            "outputs": {
                "student": scenario.get("outputs", {}).get("student", ""),
                "golden": scenario.get("outputs", {}).get("golden", "")
            }
        }

    def _score_structure(self, student: str, golden: str) -> tuple[float, Dict[str, bool]]:
        score = 25.0
        details = {}

        # Check persona / role definition
        has_persona = bool(re.search(r"\b(act as|you are a|you are an|role:|persona|as a senior|as an expert)\b", student, re.IGNORECASE))
        details["has_persona"] = has_persona
        if has_persona:
            score += 35.0

        # Check delimiters & organization (e.g., markdown, bullets, XML, quotes)
        has_delimiters = bool(re.search(r"(```|---|###|===|\n\s*[-*•]|\n\s*\d+\.|\"\"\"|<[a-zA-Z0-9]+>)", student))
        details["has_delimiters"] = has_delimiters
        if has_delimiters:
            score += 20.0

        # Check sequential step directives
        has_steps = bool(re.search(r"\b(step by step|first|then|next|finally|workflow|process)\b", student, re.IGNORECASE))
        details["has_steps"] = has_steps
        if has_steps:
            score += 20.0

        return min(100.0, score), details

    def _score_clarity(self, student: str, golden: str) -> tuple[float, Dict[str, bool]]:
        score = 30.0
        details = {}

        # Imperative action verbs at start of clauses
        has_imperatives = bool(re.search(r"\b(refactor|extract|summarize|optimize|explain|guide|generate|produce|parse|format)\b", student, re.IGNORECASE))
        details["has_imperatives"] = has_imperatives
        if has_imperatives:
            score += 35.0

        # Check for vague/uncertain language (penalize filler)
        has_vagueness = bool(re.search(r"\b(please|maybe|sort of|kind of|try to|if you can|could you|if possible)\b", student, re.IGNORECASE))
        details["has_vagueness"] = has_vagueness
        if not has_vagueness:
            score += 20.0

        # Word count sweet spot (not too short, not overly verbose)
        words = len(student.split())
        details["word_count"] = words
        if 15 <= words <= 150:
            score += 15.0
        elif words > 150:
            score += 10.0

        return min(100.0, score), details

    def _score_context(self, student: str, golden: str, scenario_key: str) -> tuple[float, Dict[str, bool]]:
        score = 25.0
        details = {}

        # Domain specific terms based on scenario
        scenario_keywords = {
            "code": [r"python", r"pep\s*8", r"complexity", r"runtime", r"docstring", r"type", r"big-o", r"o\(n\)", r"asymptotic"],
            "tutor": [r"calculus", r"socratic", r"derivative", r"math", r"speedometer", r"analogy", r"question", r"intuition"],
            "data": [r"customer", r"schema", r"json", r"urgency", r"sentiment", r"entity", r"null", r"rfc"],
            "creative": [r"executive", r"c-suite", r"kpi", r"milestone", r"board", r"quarterly", r"margin", r"cac", r"arr"]
        }

        matched_terms = 0
        patterns = scenario_keywords.get(scenario_key, scenario_keywords["code"])
        for p in patterns:
            if re.search(p, student, re.IGNORECASE):
                matched_terms += 1

        details["matched_domain_terms"] = matched_terms
        score += min(50.0, matched_terms * 16.0)

        # Target audience or context framing
        has_audience = bool(re.search(r"\b(for (the )?students|for (the )?c-suite|for developers|target|audience)\b", student, re.IGNORECASE))
        details["has_audience"] = has_audience
        if has_audience:
            score += 25.0

        return min(100.0, score), details

    def _score_completeness(self, student: str, golden: str) -> tuple[float, Dict[str, bool]]:
        score = 20.0
        details = {}

        # Explicit format schema
        has_format = bool(re.search(r"\b(json|markdown|table|bullet points|schema|format|code block|valid rfc)\b", student, re.IGNORECASE))
        details["has_format"] = has_format
        if has_format:
            score += 35.0

        # Negative constraints (what NOT to do)
        has_negative_constraints = bool(re.search(r"\b(do not|never|only|without|exclude|limit|strictly|do not include)\b", student, re.IGNORECASE))
        details["has_negative_constraints"] = has_negative_constraints
        if has_negative_constraints:
            score += 30.0

        # Edge cases & exception handling
        has_edge_cases = bool(re.search(r"\b(edge case|null|empty|exception|error|ambiguous|invalid)\b", student, re.IGNORECASE))
        details["has_edge_cases"] = has_edge_cases
        if has_edge_cases:
            score += 15.0

        return min(100.0, score), details

    def _generate_gap_analysis(self, struct: Dict, clarity: Dict, context: Dict, comp: Dict) -> List[Dict[str, str]]:
        gaps = []

        # Role persona
        if struct.get("has_persona"):
            gaps.append({
                "status": "matched",
                "title": "Role Persona Conditioning",
                "description": "Candidate assigns a defined role/persona, aligning with the expert golden prompt."
            })
        else:
            gaps.append({
                "status": "missing",
                "title": "Missing Role Persona",
                "description": "No persona specified (e.g. 'Act as a Senior Engineer...'). The LLM defaults to generic unconstrained tone."
            })

        # Output schema
        if comp.get("has_format"):
            gaps.append({
                "status": "matched",
                "title": "Output Format Contract",
                "description": "Explicit format guidelines (e.g. Markdown, JSON) provided to eliminate conversational hallucination."
            })
        else:
            gaps.append({
                "status": "missing",
                "title": "Unconstrained Output Schema",
                "description": "Lacks output format boundaries; models often respond with conversational chatter or unstructured prose."
            })

        # Negative constraints
        if comp.get("has_negative_constraints"):
            gaps.append({
                "status": "matched",
                "title": "Negative Guardrails",
                "description": "Explicit negative bounds prevent unwanted model outputs and third-party dependencies."
            })
        else:
            gaps.append({
                "status": "missing",
                "title": "Missing Negative Constraints",
                "description": "Candidate fails to define what the model must NOT do (e.g. 'Do not introduce external dependencies')."
            })

        # Domain context
        if context.get("matched_domain_terms", 0) >= 2:
            gaps.append({
                "status": "matched",
                "title": "Domain Grounding",
                "description": "Accurately specifies domain-level terminology, protocols, and architectural specifications."
            })
        else:
            gaps.append({
                "status": "partial",
                "title": "Limited Domain Context",
                "description": "Fails to ground the scenario with domain criteria (e.g. PEP 8, Big-O complexity, or RFC JSON standards)."
            })

        return gaps

    def _generate_suggestions(self, struct: Dict, clarity: Dict, context: Dict, comp: Dict) -> List[Dict[str, str]]:
        suggestions = []

        if not struct.get("has_persona"):
            suggestions.append({
                "title": "Establish an Expert Persona Anchor",
                "body": "Begin prompts with: 'You are a Senior [Domain Specialist]'. This shifts transformer attention toward technical, high-precision token spaces."
            })

        if not comp.get("has_negative_constraints"):
            suggestions.append({
                "title": "Add Boundary & Negative Constraints",
                "body": "State explicit restrictions: 'Do not import third-party packages' or 'Never reveal answers directly'. Negative bounds significantly decrease hallucinations."
            })

        if not comp.get("has_format"):
            suggestions.append({
                "title": "Enforce Strict Machine-Parsable Output",
                "body": "Specify: 'Output ONLY valid Markdown / JSON matching schema X'. This ensures deterministic, reproducible integration."
            })

        if not comp.get("has_edge_cases"):
            suggestions.append({
                "title": "Explicitly Guide Edge Case Behavior",
                "body": "Instruct how the LLM should handle null values, malformed data, or zero divisions to ensure robust solutions."
            })

        if not suggestions:
            suggestions.append({
                "title": "Benchmark Calibrated to Golden Standard",
                "body": "Your prompt satisfies major expert criteria! You can further experiment with few-shot exemplars to anchor edge cases."
            })

        return suggestions


# CLI entry point for command-line execution and batch testing
if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="NLP Prompt Engineering Assessment Engine")
    parser.add_argument("--scenario", default="code", choices=list(BENCHMARK_SCENARIOS.keys()), help="Scenario key")
    parser.add_argument("--prompt", default="Refactor this python code for better performance and add docstrings.", help="Student candidate prompt")
    args = parser.parse_args()

    evaluator = PromptEvaluator()
    scen = BENCHMARK_SCENARIOS[args.scenario]
    res = evaluator.evaluate(args.prompt, scen["golden_prompt"], scenario_key=args.scenario)

    print("\n" + "=" * 60)
    print(f" PromptEval AI: NLP Assessment Results [{scen['title']}]")
    print("=" * 60)
    print(f" Overall Prompt Quality Score: {res['overall_score']} / 100 ({res['performance_tier']})")
    print(f" Semantic Cosine Similarity:  {res['semantic_similarity']}")
    print(f" Lexical N-Gram Similarity:    {res['lexical_similarity']}")
    print("-" * 60)
    print(" Dimensional Breakdown:")
    for dim_name, dim_info in res["dimensions"].items():
        print(f"   * {dim_name.capitalize():<14}: {dim_info['score']}/100 - {dim_info['description']}")
    print("-" * 60)
    print(" Gap Analysis vs Golden Prompt:")
    for g in res["gap_analysis"]:
        print(f"   [{g['status'].upper()}] {g['title']}: {g['description']}")
    print("-" * 60)
    print(" Actionable Suggestions:")
    for s in res["suggestions"]:
        print(f"   -> {s['title']}: {s['body']}")
    print("=" * 60 + "\n")
