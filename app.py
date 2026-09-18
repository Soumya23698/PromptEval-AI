"""
PromptEval AI - Flask Backend Server
===================================
Serves the web application and provides the real-time NLP evaluation REST API
powered by Hugging Face Sentence-Transformers and PyTorch.
"""

import os
import sys
import webbrowser
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from nlp_engine import PromptEvaluator, BENCHMARK_SCENARIOS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

# Initialize NLP model at startup
print("[PromptEval AI] Initializing Sentence-Transformers NLP Engine on PyTorch...")
evaluator = PromptEvaluator()
print("[PromptEval AI] NLP Model loaded and ready!")


@app.route("/")
def index():
    """Serve main web page interface."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint confirming backend model status."""
    return jsonify({
        "status": "online",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "framework": "PyTorch + Hugging Face",
        "scenarios_available": list(BENCHMARK_SCENARIOS.keys())
    })


@app.route("/api/scenarios", methods=["GET"])
def get_scenarios():
    """Retrieve available task scenarios and golden prompt benchmarks."""
    return jsonify({
        "status": "success",
        "scenarios": BENCHMARK_SCENARIOS
    })


@app.route("/api/assess", methods=["POST"])
def assess_prompt():
    """
    NLP Assessment Endpoint
    Takes a candidate prompt and scenario, evaluates it against the expert golden prompt,
    and returns dense semantic similarity, dimensional scores, gap analysis, and suggestions.
    """
    data = request.get_json(force=True, silent=True) or {}
    student_prompt = data.get("student_prompt", "").strip()
    scenario_key = data.get("scenario", "code")
    golden_prompt = data.get("golden_prompt", "").strip()

    if not student_prompt:
        return jsonify({
            "status": "error",
            "message": "student_prompt parameter is required."
        }), 400

    scen = BENCHMARK_SCENARIOS.get(scenario_key, BENCHMARK_SCENARIOS["code"])
    if not golden_prompt:
        golden_prompt = scen["golden_prompt"]

    try:
        results = evaluator.evaluate(
            student_prompt=student_prompt,
            golden_prompt=golden_prompt,
            scenario_key=scenario_key,
            dynamic_golden=True
        )
        return jsonify({
            "status": "success",
            "source": "pytorch_sentence_transformer",
            "results": results
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/generate_golden_prompt", methods=["POST"])
def generate_golden_prompt():
    """
    Synthesize an Expert-Designed Golden Prompt tailored directly to a candidate input prompt.
    """
    data = request.get_json(force=True, silent=True) or {}
    student_prompt = data.get("student_prompt", "").strip()
    scenario_key = data.get("scenario", "code")

    if not student_prompt:
        return jsonify({
            "status": "error",
            "message": "student_prompt parameter is required."
        }), 400

    try:
        expert_golden = evaluator.generate_expert_golden_prompt(student_prompt, scenario_key)
        return jsonify({
            "status": "success",
            "expert_golden_prompt": expert_golden
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


def open_browser(port):
    """Open user default browser after a short delay."""
    url = f"http://localhost:{port}"
    print(f"[PromptEval AI] Opening browser at {url}...")
    webbrowser.open(url)


if __name__ == "__main__":
    PORT = 5000
    print("\n" + "=" * 60)
    print(" PromptEval AI: NLP Model Web Server Running")
    print(f" Web Interface: http://localhost:{PORT}")
    print(f" REST API:      http://localhost:{PORT}/api/assess")
    print(" Backend Model: PyTorch + Hugging Face Sentence-Transformers")
    print(" Press Ctrl+C in this terminal to shut down.")
    print("=" * 60 + "\n")

    # Launch browser in separate thread
    threading.Timer(1.2, open_browser, args=[PORT]).start()

    # Start Flask server
    app.run(host="0.0.0.0", port=PORT, debug=False)
