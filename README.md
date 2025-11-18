# Python Backend untuk SciFind

## Setup

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Pastikan file `indexed_data.json` ada di folder backend dengan format:
```json
[
  {
    "id": "1",
    "judul": "The Meg",
    "poster": "https://...",
    "content": "Full text content...",
    "isi": "Preview text..."
  }
]
```

3. Jalankan server:
```bash
python app.py
```

Server akan berjalan di `http://localhost:5000`

## API Endpoints

### 1. Search
- **POST** `/api/search`
- **GET** `/api/search?query=venom&method=hybrid&top_k=10`

Body (POST):
```json
{
  "query": "alien movie",
  "method": "hybrid",
  "top_k": 10
}
```

Response:
```json
{
  "query": "alien movie",
  "method": "hybrid",
  "total_results": 5,
  "results": [...]
}
```

### 2. Health Check
- **GET** `/api/health`

### 3. Get Document
- **GET** `/api/document/<doc_id>`

## Methods
- `tfidf`: Menggunakan TF-IDF + Cosine Similarity
- `jaccard`: Menggunakan Jaccard Similarity
- `hybrid`: Kombinasi TF-IDF (70%) + Jaccard (30%)
