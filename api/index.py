from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from difflib import get_close_matches
from collections import Counter
import os

app = Flask(__name__)
# Allow CORS for frontend domains
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:5173",
            "http://localhost:5000", 
            "https://*.vercel.app"
        ]
    }
})

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# Load data
print("Loading data...")
clean_docs_path = os.path.join(PARENT_DIR, "clean_documents.json")
tfidf_index_path = os.path.join(PARENT_DIR, "tfidf_index.json")

with open(clean_docs_path, "r", encoding='utf-8') as f:
    CLEAN_DOCS = json.load(f)

with open(tfidf_index_path, "r", encoding='utf-8') as f:
    TFIDF = json.load(f)

tfidf_matrix = np.array(TFIDF["matrix"])
vocab = TFIDF["vocab"]
vocab_index = {w: i for i, w in enumerate(vocab)}
idf = np.array(TFIDF["idf"])
filenames = TFIDF["filenames"]

def preprocess_query(q):
    q = q.lower().replace("-", " ")    
    return q.split()

def compute_tf(tokens):
    vec = np.zeros(len(vocab_index))
    count = Counter(tokens)
    for word, freq in count.items():
        if word in vocab_index:
            vec[vocab_index[word]] = freq
    return vec

def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def autocorrect_safe(tokens):
    corrected = []
    suggestions = []
    for t in tokens:
        matches = get_close_matches(t, vocab, n=1, cutoff=0.75)
        if matches and matches[0] != t:
            suggestions.append({"original": t, "suggestion": matches[0]})
            corrected.append(t)   
        else:
            corrected.append(t)
    return corrected, suggestions

title_map = { d["title"]: d for d in CLEAN_DOCS }

def jaccard_similarity(q_tokens, doc_tokens, title_tokens):
    if len(q_tokens) == 0:
        return 0.0
    q_set = set(q_tokens)
    d_set = set(doc_tokens)
    inter = len(q_set & d_set)
    union = len(q_set | d_set)
    base = inter / union if union > 0 else 0.0
    if len(q_set & set(title_tokens)) > 0:
        base += 0.3
    return base

def search(query, method="hybrid", top_k=10):
    q_tokens = preprocess_query(query)
    q_tokens, suggestions = autocorrect_safe(q_tokens)
    
    # TF-IDF
    q_tf = compute_tf(q_tokens)
    q_vec = q_tf * idf
    tfidf_scores = []
    
    for i, doc_vec in enumerate(tfidf_matrix):
        sim = cosine_similarity(q_vec, doc_vec)
        title = filenames[i].lower().replace("-", " ")
        title_tokens = title.split()
        if len(set(q_tokens) & set(title_tokens)) > 0:
            sim += 0.3
        tfidf_scores.append((i, sim))
    
    tfidf_ranked = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    tfidf_ranked = [(i, s) for (i, s) in tfidf_ranked if s > 0]
    
    # Jaccard
    jaccard_scores = []
    for doc in CLEAN_DOCS:
        sim = jaccard_similarity(
            q_tokens,
            doc["tokens"],
            doc["title"].lower().replace("-", " ").split()
        )
        jaccard_scores.append((doc["title"], sim))
    
    jaccard_ranked = sorted(jaccard_scores, key=lambda x: x[1], reverse=True)
    jaccard_ranked = [(t, s) for (t, s) in jaccard_ranked if s > 0]
    
    # Build results
    result_tfidf = []
    for idx, score in tfidf_ranked[:top_k]:
        title = filenames[idx]
        if title in title_map:
            d = title_map[title]
            result_tfidf.append({
                "judul": d["title"],
                "poster": d["poster"],
                "isi": d["description"][:500] + "...",
                "content": d["description"],
                "score": float(score),
                "tfidf_score": float(score),
                "jaccard_score": 0,
                "method": "TF-IDF"
            })
    
    result_jaccard = []
    for title, score in jaccard_ranked[:top_k]:
        if title in title_map:
            d = title_map[title]
            result_jaccard.append({
                "judul": d["title"],
                "poster": d["poster"],
                "isi": d["description"][:500] + "...",
                "content": d["description"],
                "score": float(score),
                "tfidf_score": 0,
                "jaccard_score": float(score),
                "method": "Jaccard"
            })
    
    if method == "hybrid":
        combined = {}
        for item in result_tfidf:
            title = item["judul"]
            combined[title] = {
                "data": item,
                "tfidf_score": item["score"],
                "jaccard_score": 0
            }
        for item in result_jaccard:
            title = item["judul"]
            if title in combined:
                combined[title]["jaccard_score"] = item["score"]
            else:
                combined[title] = {
                    "data": item,
                    "tfidf_score": 0,
                    "jaccard_score": item["score"]
                }
        
        hybrid_results = []
        for title, scores in combined.items():
            hybrid_score = (scores["tfidf_score"] * 0.7 + scores["jaccard_score"] * 0.3)
            result = scores["data"].copy()
            result["score"] = float(hybrid_score)
            result["method"] = "Hybrid"
            result["tfidf_score"] = float(scores["tfidf_score"])
            result["jaccard_score"] = float(scores["jaccard_score"])
            hybrid_results.append(result)
        
        hybrid_results.sort(key=lambda x: x["score"], reverse=True)
        return {
            "results": hybrid_results[:top_k],
            "suggestions": suggestions,
            "total": len(hybrid_results)
        }
    elif method == "tfidf":
        return {
            "results": result_tfidf,
            "suggestions": suggestions,
            "total": len(result_tfidf)
        }
    elif method == "jaccard":
        return {
            "results": result_jaccard,
            "suggestions": suggestions,
            "total": len(result_jaccard)
        }

@app.route('/api/search', methods=['GET', 'POST'])
def api_search():
    if request.method == 'POST':
        data = request.get_json()
        query = data.get('query', '')
        method = data.get('method', 'hybrid')
        top_k = data.get('top_k', 10)
    else:
        query = request.args.get('query', '')
        method = request.args.get('method', 'hybrid')
        top_k = int(request.args.get('top_k', 10))
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        result = search(query, method, top_k)
        return jsonify({
            'query': query,
            'method': method,
            'total_results': result['total'],
            'results': result['results'],
            'suggestions': result['suggestions']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'total_documents': len(CLEAN_DOCS),
        'vocabulary_size': len(vocab)
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'SciFind Search API',
        'endpoints': ['/api/search', '/api/health']
    })

# Vercel serverless function handler
def handler(request):
    with app.request_context(request.environ):
        return app.full_dispatch_request()
