from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from collections import Counter
import os

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "http://localhost:5000", "https://*.vercel.app"]
    }
})

# Global cache for data
_data_cache = {}

def load_data():
    """Lazy load data only when needed"""
    if not _data_cache:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        with open(os.path.join(parent_dir, "clean_documents.json"), "r", encoding="utf-8") as f:
            _data_cache["documents"] = json.load(f)
        
        with open(os.path.join(parent_dir, "tfidf_index.json"), "r", encoding="utf-8") as f:
            tfidf = json.load(f)
            _data_cache["tfidf_matrix"] = np.array(tfidf["matrix"])
            _data_cache["vocabulary"] = tfidf["vocab"]
            _data_cache["idf_values"] = np.array(tfidf["idf"])
            _data_cache["filenames"] = tfidf["filenames"]
    
    return _data_cache

def preprocess_query(query):
    return query.lower().split()

def compute_tf(tokens):
    tf = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in tf.items()}

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def search_tfidf(query, top_k=10):
    data = load_data()
    query_tokens = preprocess_query(query)
    query_tf = compute_tf(query_tokens)
    
    query_vector = np.zeros(len(data["vocabulary"]))
    for word, tf_val in query_tf.items():
        if word in data["vocabulary"]:
            idx = data["vocabulary"].index(word)
            query_vector[idx] = tf_val * data["idf_values"][idx]
    
    scores = []
    for i, doc_vec in enumerate(data["tfidf_matrix"]):
        score = cosine_similarity(query_vector, np.array(doc_vec))
        if score > 0:
            filename = data["filenames"][i]
            doc = next((d for d in data["documents"] if d["title"] == filename), None)
            if doc:
                scores.append({"doc": doc, "score": float(score)})
    
    return sorted(scores, key=lambda x: x["score"], reverse=True)[:top_k]

def search_jaccard(query, top_k=10):
    data = load_data()
    query_tokens = set(preprocess_query(query))
    
    scores = []
    for doc in data["documents"]:
        doc_tokens = set(doc.get("tokens", []))
        score = jaccard_similarity(query_tokens, doc_tokens)
        if score > 0:
            scores.append({"doc": doc, "score": float(score)})
    
    return sorted(scores, key=lambda x: x["score"], reverse=True)[:top_k]

def search_hybrid(query, top_k=10):
    tfidf_results = search_tfidf(query, top_k * 2)
    jaccard_results = search_jaccard(query, top_k * 2)
    
    combined = {}
    for item in tfidf_results:
        title = item["doc"]["title"]
        combined[title] = {
            "doc": item["doc"],
            "tfidf_score": item["score"],
            "jaccard_score": 0.0
        }
    
    for item in jaccard_results:
        title = item["doc"]["title"]
        if title in combined:
            combined[title]["jaccard_score"] = item["score"]
        else:
            combined[title] = {
                "doc": item["doc"],
                "tfidf_score": 0.0,
                "jaccard_score": item["score"]
            }
    
    results = []
    for title, data in combined.items():
        hybrid_score = 0.7 * data["tfidf_score"] + 0.3 * data["jaccard_score"]
        results.append({
            "title": data["doc"]["title"],
            "poster": data["doc"].get("poster", ""),
            "description": data["doc"].get("description", ""),
            "score": float(hybrid_score),
            "tfidf_score": float(data["tfidf_score"]),
            "jaccard_score": float(data["jaccard_score"])
        })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

@app.route("/api/search", methods=["POST", "GET"])
def search():
    if request.method == "GET":
        query = request.args.get("query", "")
        method = request.args.get("method", "hybrid")
        top_k = int(request.args.get("top_k", 10))
    else:
        data = request.get_json()
        query = data.get("query", "")
        method = data.get("method", "hybrid")
        top_k = data.get("top_k", 10)
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    if method == "tfidf":
        results = search_tfidf(query, top_k)
        formatted = [{"title": r["doc"]["title"], "poster": r["doc"].get("poster", ""),
                     "description": r["doc"].get("description", ""), "tfidf_score": r["score"]}
                    for r in results]
    elif method == "jaccard":
        results = search_jaccard(query, top_k)
        formatted = [{"title": r["doc"]["title"], "poster": r["doc"].get("poster", ""),
                     "description": r["doc"].get("description", ""), "jaccard_score": r["score"]}
                    for r in results]
    else:
        formatted = search_hybrid(query, top_k)
    
    return jsonify({"results": formatted, "query": query, "method": method})

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})
