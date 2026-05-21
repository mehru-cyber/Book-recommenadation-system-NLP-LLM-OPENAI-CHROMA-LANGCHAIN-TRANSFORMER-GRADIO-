# 📚 Semantic Book Recommender

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-black?style=flat-square)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings-412991?style=flat-square&logo=openai)](https://openai.com)
[![Gradio](https://img.shields.io/badge/Gradio-5.9-orange?style=flat-square)](https://gradio.app)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--store-blueviolet?style=flat-square)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> **Find your next book using natural language** — describe what you want to read, filter by genre and emotional tone, and get AI-powered recommendations from a library of 7,000+ books.

![Dashboard preview](cover-not-found.jpg)

---

## ✨ What makes this different from a search bar

Traditional book search requires you to know what you're looking for. This system understands *meaning*. You can type:

> *"a story about a person seeking revenge in a dystopian world"*

…and it finds books that match that concept — even if none of them contain those exact words. It then lets you filter by **Fiction / Nonfiction** and sort by **emotional tone** (joyful, suspenseful, sad, surprising, or angry) using LLM-extracted sentiment.

---

## 🎬 Demo

```
Query:      "a book about forgiveness and second chances"
Category:   Fiction
Tone:       Sad

→ Returns 16 book covers with titles, authors, and descriptions
  sorted by sadness score extracted from their descriptions
```

> 🚀 Run it yourself in under 5 minutes — see [Quickstart](#-quickstart)

---

## 🏗 System architecture

```
7,000+ books (Kaggle)
        │
        ▼
┌─────────────────────┐
│  data-exploration   │  Clean missing values, filter short descriptions,
│      .ipynb         │  tag each book with its ISBN13
└────────┬────────────┘
         │  books_cleaned.csv
         ▼
┌─────────────────────┐     ┌──────────────────────────┐
│ text-classification │     │   sentiment-analysis     │
│      .ipynb         │     │        .ipynb            │
│                     │     │                          │
│ facebook/bart-large │     │ j-hartmann/emotion-      │
│ -mnli (zero-shot)   │     │ distilroberta-base       │
│                     │     │                          │
│ Fiction / Nonfiction│     │ joy, fear, anger,        │
│ classification      │     │ sadness, surprise scores │
└────────┬────────────┘     └────────────┬─────────────┘
         │                               │
         └──────────┬────────────────────┘
                    │  books_with_emotions.csv
                    ▼
        ┌───────────────────────┐
        │    vector-search      │
        │       .ipynb          │
        │                       │
        │  OpenAI Embeddings    │
        │  + ChromaDB           │
        │  semantic index of    │
        │  all book descriptions│
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  gradio-dashboard.py  │  ← User-facing app
        │                       │
        │  Query → vector search│
        │  → filter by category │
        │  → sort by emotion    │
        │  → display 16 books   │
        └───────────────────────┘
```

---

## 📓 Notebook breakdown

| Notebook | Purpose | Key techniques |
|---|---|---|
| `data-exploration.ipynb` | Data cleaning & EDA | Missing value heatmaps, description length filtering (≥25 words), ISBN tagging |
| `text-classification.ipynb` | Genre labeling | Zero-shot classification with `facebook/bart-large-mnli`, maps 40+ raw categories → Fiction / Nonfiction |
| `sentiment-analysis.ipynb` | Emotion extraction | Sentence-level emotion scoring with `j-hartmann/emotion-english-distilroberta-base`, max-pooling across sentences |
| `vector-search.ipynb` | Semantic search index | OpenAI embeddings + ChromaDB, tagged descriptions as documents, similarity search at query time |

---

## 🚀 Quickstart

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Install & run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/semantic-book-recommender
cd semantic-book-recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# 4. Run the notebooks in order (one-time setup)
jupyter notebook data-exploration.ipynb       # → books_cleaned.csv
jupyter notebook text-classification.ipynb    # → books_with_categories.csv
jupyter notebook sentiment-analysis.ipynb     # → books_with_emotions.csv
jupyter notebook vector-search.ipynb          # → tagged_description.txt + ChromaDB index

# 5. Launch the dashboard
python gradio-dashboard.py
```

Open `http://localhost:7860` in your browser.

> ⚠️ Steps 2–4 are one-time setup. Once the CSVs and vector index are built, you only need step 5 to use the app.

---

## 📁 Project structure

```
semantic-book-recommender/
├── data-exploration.ipynb        # Step 1 — clean & prepare raw data
├── text-classification.ipynb     # Step 2 — genre classification
├── sentiment-analysis.ipynb      # Step 3 — emotion scoring
├── vector-search.ipynb           # Step 4 — build semantic search index
├── gradio-dashboard.py           # Step 5 — Gradio web app
├── cover-not-found.jpg           # Fallback cover image
├── requirements.txt
└── .env                          # Your OPENAI_API_KEY (not committed)

# Generated files (created by running notebooks):
├── books_cleaned.csv             # After step 1
├── books_with_categories.csv     # After step 2
├── books_with_emotions.csv       # After step 3
└── tagged_description.txt        # After step 4
```

---

## 🧠 How each piece works

### Semantic search
Book descriptions are embedded using OpenAI's `text-embedding-ada-002` model and stored in ChromaDB. At query time, the user's natural language input is embedded with the same model and cosine similarity retrieves the top 50 closest books before filtering and re-ranking.

### Zero-shot genre classification
Instead of training a custom classifier (which would require labeled data), `facebook/bart-large-mnli` classifies each book description as Fiction or Nonfiction using natural language inference — no fine-tuning required.

### Emotion scoring
Each book description is split into sentences. `j-hartmann/emotion-english-distilroberta-base` scores each sentence across 7 emotions (joy, fear, anger, sadness, surprise, disgust, neutral). The max score per emotion across all sentences is kept, capturing the *peak* emotional intensity in the description.

### Gradio dashboard
The app combines all three signals: vector similarity (relevance), category filter (genre), and emotion sort (tone). The result is a 4×2 gallery of book covers with author and description previews.

---

## 🛠 Tech stack

| Layer | Tool |
|---|---|
| Data wrangling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| NLP / classification | Hugging Face Transformers (`facebook/bart-large-mnli`, `j-hartmann/emotion-english-distilroberta-base`) |
| Embeddings | OpenAI `text-embedding-ada-002` via LangChain |
| Vector store | ChromaDB via LangChain |
| Web app | Gradio 5.9 |
| Dataset | [7k Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) (Kaggle) |

---








