
from pathlib import Path
import re

import faiss
import numpy as np
import pandas as pd
import streamlit as st

# Project paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MOVIES_PATH = ARTIFACTS_DIR / "movies.csv"
MOVIE_IDS_PATH = ARTIFACTS_DIR / "movie_ids_sorted.npy"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "X_norm.npy"
INDEX_PATH = ARTIFACTS_DIR / "hnsw_full.index"


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Movie Recommender")
st.write(
    "Discover similar movies using SVD embeddings "
    "and FAISS HNSW approximate nearest-neighbor search."
)


def normalize_title(title):
    """Normalize MovieLens titles, including 'Matrix, The'."""
    title = re.sub(r"\s*\(\d{4}\)$", "", str(title).strip())

    match = re.match(r"^(.*),\s*(The|A|An)$", title, flags=re.I)
    if match:
        title = f"{match.group(2)} {match.group(1)}"

    return " ".join(title.casefold().split())


@st.cache_resource
def load_artifacts():
    movies = pd.read_csv(MOVIES_PATH)
    movie_ids = np.load(MOVIE_IDS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    index = faiss.read_index(str(INDEX_PATH))

    # Keep metadata for movies present in the index
    movies["movieId"] = movies["movieId"].astype(int)
    movies["normalized_title"] = movies["title"].apply(normalize_title)

    indexed_ids = set(map(int, movie_ids))
    movies = movies[movies["movieId"].isin(indexed_ids)].copy()

    # Map movie ID to its vector position
    movie_id_to_idx = {
        int(movie_id): idx
        for idx, movie_id in enumerate(movie_ids)
    }

    return movies, movie_ids, embeddings, index, movie_id_to_idx


try:
    movies, movie_ids, embeddings, index, movie_id_to_idx = load_artifacts()
except Exception as exc:
    st.error(f"Could not load project artifacts: {exc}")
    st.info(
        "Check that artifacts contains movies.csv, movie_ids_sorted.npy, "
        "X_norm.npy, and hnsw_full.index."
    )
    st.stop()


def find_movie(query):
    """Find an exact normalized title before attempting partial matching."""
    normalized_query = normalize_title(query)

    exact = movies[movies["normalized_title"] == normalized_query]
    if not exact.empty:
        return exact.iloc[0], None

    partial = movies[
        movies["normalized_title"].str.contains(
            normalized_query, regex=False, na=False
        )
    ]

    if len(partial) == 1:
        return partial.iloc[0], None

    if len(partial) > 1:
        choices = partial["title"].sort_values().tolist()
        return None, choices

    return None, []


def recommend_movies(movie_id, top_n=10):
    """Retrieve similar movies from the HNSW index."""
    query_idx = movie_id_to_idx.get(int(movie_id))

    if query_idx is None:
        return pd.DataFrame()

    query = np.ascontiguousarray(
        embeddings[query_idx:query_idx + 1],
        dtype=np.float32,
    )

    # Search for one extra result so the query movie can be removed
    scores, indices = index.search(query, top_n + 1)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx == query_idx:
            continue

        neighbor_id = int(movie_ids[idx])
        match = movies[movies["movieId"] == neighbor_id]

        if match.empty:
            continue

        row = match.iloc[0]
        results.append({
            "Movie": row["title"],
            "Genres": row["genres"],
            "Cosine similarity": round(float(score), 4),
        })

        if len(results) >= top_n:
            break

    return pd.DataFrame(results)


st.caption(
    f"Loaded {len(movie_ids):,} indexed movies · "
    f"{embeddings.shape[1]}-dimensional embeddings"
)

movie_query = st.text_input(
    "Enter a movie title",
    placeholder="e.g. The Matrix or Toy Story",
)

top_n = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Get recommendations", type="primary"):
    if not movie_query.strip():
        st.warning("Enter a movie title first.")
    else:
        movie, choices = find_movie(movie_query)

        if movie is None and choices:
            st.warning("Several titles match. Choose the exact movie:")
            selected_title = st.selectbox("Matching movies", choices)

            if st.button("Use selected movie"):
                selected = movies[movies["title"] == selected_title].iloc[0]
                result = recommend_movies(selected["movieId"], top_n)
                st.subheader(f"Movies similar to {selected_title}")
                st.dataframe(result, use_container_width=True, hide_index=True)

        elif movie is None:
            st.error("No matching movie found. Try a different title.")
        else:
            result = recommend_movies(movie["movieId"], top_n)
            st.subheader(f"Movies similar to {movie['title']}")

            if result.empty:
                st.info("No recommendations were found.")
            else:
                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True,
                )

st.divider()
st.caption(
    "Recall@10 measures agreement with exact nearest-neighbor search, "
    "not real-world recommendation quality."
)
