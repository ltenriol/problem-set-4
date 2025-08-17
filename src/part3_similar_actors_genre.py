"""
Find similar actors by genre profile.

- Builds an actor×genre matrix from a JSONL file where each line is one movie
  with keys: 'actors' -> list of [actor_id, actor_name], 'genres' -> list[str]
- Uses Chris Hemsworth (nm1165110) as the query actor
- Computes distances:
    * cosine (for CSV output)
    * euclidean (for comparison/printout)
- Outputs top-10 most similar by cosine distance to /data with a timestamped name
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import cosine_distances


def build_actor_genre_matrix(jsonl_path: str) -> pd.DataFrame:
    """
    Build a DataFrame: rows = actors, cols = genres, values = #appearances.

    Parameters
    ----------
    jsonl_path : str
        Path to JSONL with one movie per line. Each movie must contain:
        - 'actors': list of [actor_id, actor_name]
        - 'genres': list of genre strings

    Returns
    -------
    pd.DataFrame
        Pivot table indexed by ['actor_id','actor_name'], columns=genres, values=int
    """
    records = []
    with open('imdb_movies_data.jsonl', "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            movie = json.loads(line)
            genres = movie.get("genres", []) or []
            actors = movie.get("actors", []) or []
            for actor_id, actor_name in actors:
                for g in genres:
                    records.append(
                        {"actor_id": actor_id, "actor_name": actor_name, "genre": g}
                    )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No actor/genre data found. Check the input JSONL file.")

    pivot = (
        df.groupby(["actor_id", "actor_name", "genre"])
          .size()
          .reset_index(name="count")
          .pivot(index=["actor_id", "actor_name"], columns="genre", values="count")
          .fillna(0)
          .astype(int)
    )
    return pivot


def top_similar_actors(
    actor_genre: pd.DataFrame, query_actor_id: str, top_n: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute cosine & euclidean distances from query to all other actors.

    Returns
    -------
    results : pd.DataFrame
        Every actor with both distances.
    top10_cosine : pd.DataFrame
        Top-N by cosine distance (ascending; smaller = more similar).
    top10_euclid : pd.DataFrame
        Top-N by euclidean distance (ascending).
    """
    # Separate features and index pieces
    features = actor_genre.values  # (n_actors, n_genres)
    ids = actor_genre.index.get_level_values("actor_id").to_numpy()
    names = actor_genre.index.get_level_values("actor_name").to_numpy()

    # Locate query row
    matches = np.where(ids == query_actor_id)[0]
    if matches.size == 0:
        raise ValueError(f"Query actor_id {query_actor_id} not found.")
    q_idx = int(matches[0])

    # Ensure 2D query vector (fixes the inconsistent samples error)
    query_vec = features[q_idx].reshape(1, -1)

    # Distances
    euclid_metric = DistanceMetric.get_metric("euclidean")
    euclid = euclid_metric.pairwise(query_vec, features).flatten()  # shape (n_actors,)
    cos = cosine_distances(query_vec, features).flatten()

    # Assemble results
    results = pd.DataFrame(
        {"actor_id": ids, "actor_name": names, "euclidean": euclid, "cosine": cos}
    )

    # Exclude the query actor from ranking
    results = results[results["actor_id"] != query_actor_id].copy()

    # Top lists
    top10_cosine = results.sort_values("cosine", ascending=True).head(top_n)
    top10_euclid = results.sort_values("euclidean", ascending=True).head(top_n)

    return results, top10_cosine, top10_euclid


def main():
    # --- Paths & query ---
    jsonl_path = "data/imdb_movies_data.json1"
    query_id = "nm1165110"   # Chris Hemsworth

    # --- Build feature matrix ---
    actor_genre = build_actor_genre_matrix(jsonl_path)

    # --- Compute similarities ---
    results, top10_cosine, top10_euclid = top_similar_actors(actor_genre, query_id, 10)

    # --- Save top-10 by cosine to /data ---
    os.makedirs("/data", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"/data/similar_actors_genre_{ts}.csv"
    top10_cosine.to_csv(out_csv, index=False)

    # --- Print summaries ---
    print(f"\nTop 10 most similar to Chris Hemsworth (cosine distance) → {out_csv}")
    print(top10_cosine[["actor_id", "actor_name", "cosine"]])

    print("\nTop 10 by Euclidean distance (for comparison):")
    print(top10_euclid[["actor_id", "actor_name", "euclidean"]])

    # Explain difference
    print(
        "\nObservation:\n"
        "- Cosine distance focuses on the *proportions* of genres (direction). "
        "Actors with a similar genre mix rank highest even if their total number "
        "of films differs.\n"
        "- Euclidean distance is sensitive to *counts* (magnitude). "
        "Actors with more/fewer appearances can move up or down even with a similar mix.\n"
        "So the Euclidean top-10 can differ from the cosine top-10 when actors have "
        "different overall film counts across the same genres."
    )


if __name__ == "__main__":
    main()
