# Movie Recommender

A Streamlit application that recommends movies similar to a selected title. It
uses precomputed SVD embeddings and a FAISS HNSW approximate nearest-neighbor
index to retrieve similar movies quickly.

## Features

- Search by exact or partial movie title.
- Handles common MovieLens title formatting such as `Matrix, The (1999)`.
- Choose how many recommendations to display (5-20).
- View each recommendation's title, genres, and cosine similarity score.
- Resolve ambiguous searches by selecting from matching titles.

## Requirements

- Python 3.9 or newer
- The precomputed files in the [`artifacts`](./artifacts) directory:
  - `movies.csv`
  - `movie_ids_sorted.npy`
  - `X_norm.npy`
  - `hnsw_full.index`

The artifact files are loaded at startup and are required to run the app.

## Setup

From the project directory, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

If PowerShell prevents script activation, run the app with the virtual
environment's executable directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the application

With the virtual environment activated:

```powershell
streamlit run app.py
```

Or, without activating the environment:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Streamlit will print a local URL, normally
`http://localhost:8501`, in the terminal.

## Usage

1. Enter a movie title, such as `The Matrix` or `Toy Story`.
2. Set the number of recommendations with the slider.
3. Select **Get recommendations**.
4. If multiple titles match, choose the intended title and select
   **Use selected movie**.

## Project structure

```text
.
├── app.py                 # Streamlit user interface and recommendation logic
├── requirements.txt       # Python dependencies
└── artifacts/
    ├── movies.csv         # Movie metadata
    ├── movie_ids_sorted.npy
    ├── X_norm.npy         # Normalized movie embeddings
    └── hnsw_full.index    # FAISS HNSW search index
```

At startup, the app loads the metadata, embeddings, movie ID mapping, and FAISS
index. A selected movie's embedding is then queried against the index; the
selected movie itself is removed from the results before the top recommendations
are displayed.

## Notes

- Recommendation quality depends on the precomputed embeddings and index.
- The displayed cosine similarity is a nearest-neighbor similarity score, not
  a rating or probability that a user will enjoy a movie.
- The app reports an artifact-loading error in the interface if any required
  file is missing or cannot be read.