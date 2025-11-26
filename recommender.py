"""
Unified Recommender System (Runnable)

Includes:
 - Synthetic-user generation (Option B: random synthetic users)
 - User-based Collaborative Filtering
 - Item-based Collaborative Filtering
 - SVD (TruncatedSVD) reconstructed predictions
 - NMF reconstructed predictions
 - Content-based (TF-IDF) movie-to-movie
 - Hybrid movie-to-movie (content + numeric)
 - USER INPUT for content + hybrid recommendations

How to run:
 - Put imdb-movies-dataset.csv in the same folder
 - Activate venv and install: pip install pandas numpy scikit-learn
 - python recommender_full.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------
# 0. Utility / cleaning helpers
# ---------------------------------------------------------
def safe_read_csv(path):
    df = pd.read_csv(path)
    return df

def clean_numeric_cols(df, num_features):
    for col in num_features:
        if col not in df.columns:
            df[col] = np.nan

        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = df[col].str.replace("min", "", regex=False)
        df[col] = df[col].str.replace("-", "", regex=False)
        df[col] = df[col].str.replace("N/A", "", regex=False)
        df[col] = df[col].str.strip()

        df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[col].notna().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(0.0)

    return df


# ---------------------------------------------------------
# 1. Load & prepare movies dataframe
# ---------------------------------------------------------
def load_movies(path="imdb-movies-dataset.csv"):
    df = safe_read_csv(path)

    required = [
        'Poster','Title','Year','Certificate','Duration (min)','Genre','Rating','Metascore',
        'Director','Cast','Votes','Description','Review Count','Review Title'
    ]

    existing = [c for c in required if c in df.columns]
    df = df[existing].copy()

    df = df.dropna(subset=['Title']).reset_index(drop=True)

    # Fill text columns
    text_cols = ['Genre', 'Director', 'Cast', 'Description', 'Review Title', 'Certificate']
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
        else:
            df[c] = ""

    num_features = ['Rating','Metascore','Duration (min)','Votes','Review Count','Year']
    df = clean_numeric_cols(df, num_features)

    df = df.drop_duplicates(subset=['Title']).reset_index(drop=True)

    df["metadata"] = (
        df["Genre"].astype(str) + " " +
        df["Director"].astype(str) + " " +
        df["Cast"].astype(str) + " " +
        df["Description"].astype(str) + " " +
        df.get("Review Title", "").astype(str) + " " +
        df.get("Certificate", "").astype(str)
    )

    return df


# ---------------------------------------------------------
# 2. Generate synthetic user ratings
# ---------------------------------------------------------
def generate_synthetic_ratings(movies_df, n_users=120, min_movies_per_user=20, max_movies_per_user=150, seed=RANDOM_SEED):
    np.random.seed(seed)
    movies = movies_df['Title'].values
    n_movies = len(movies)
    rows = []

    for user in range(1, n_users + 1):
        k = int(np.clip(np.random.poisson(50), min_movies_per_user, min(max_movies_per_user, n_movies)))
        chosen = np.random.choice(movies, size=min(k, n_movies), replace=False)
        for title in chosen:
            base_row = movies_df.loc[movies_df['Title'] == title]
            if not base_row.empty and base_row.iloc[0]['Rating'] != 0:
                base = float(base_row.iloc[0]['Rating'])
                rating = np.clip(base + np.random.normal(scale=1.2), 1, 10)
            else:
                rating = np.clip(np.random.normal(loc=6.5, scale=1.8), 1, 10)
            rows.append({'user_id': int(user), 'Title': title, 'Rating': float(round(rating, 2))})

    ratings_df = pd.DataFrame(rows)
    return ratings_df


# ---------------------------------------------------------
# 3. User-item matrix
# ---------------------------------------------------------
def build_user_item_matrix(ratings_df, movies_index):
    uim = ratings_df.pivot_table(index='user_id', columns='Title', values='Rating').fillna(0)

    for title in movies_index:
        if title not in uim.columns:
            uim[title] = 0.0

    uim = uim.reindex(columns=list(movies_index)).sort_index()
    return uim


# ---------------------------------------------------------
# 4. SVD & NMF predictions
# ---------------------------------------------------------
def compute_svd_predictions(user_item_matrix, n_components=20, random_state=RANDOM_SEED):
    n_components = min(n_components, min(user_item_matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    user_features = svd.fit_transform(user_item_matrix)
    item_features = svd.components_
    pred_matrix = np.dot(user_features, item_features)
    pred = pd.DataFrame(pred_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)
    return pred, svd

def compute_nmf_predictions(user_item_matrix, n_components=20, random_state=RANDOM_SEED, max_iter=1000):
    n_components = min(n_components, min(user_item_matrix.shape) - 1)
    nmf = NMF(n_components=n_components, init='nndsvda', random_state=random_state, max_iter=max_iter)
    W = nmf.fit_transform(user_item_matrix.clip(lower=0))
    H = nmf.components_
    pred_matrix = np.dot(W, H)
    pred = pd.DataFrame(pred_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)
    return pred, nmf


# ---------------------------------------------------------
# 5. Similarities
# ---------------------------------------------------------
def item_similarity_from_user_item(user_item_matrix):
    sim = cosine_similarity(user_item_matrix.T)
    return pd.DataFrame(sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def compute_content_similarity(movies_df, max_features=5000):
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(movies_df['metadata'].values)
    sim = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(sim, index=movies_df['Title'].values, columns=movies_df['Title'].values)

def compute_numeric_similarity(movies_df, num_features=['Rating','Metascore','Duration (min)','Votes','Review Count','Year']):
    scaler = MinMaxScaler()
    num_scaled = scaler.fit_transform(movies_df[num_features].astype(float).values)
    sim = cosine_similarity(num_scaled)
    return pd.DataFrame(sim, index=movies_df['Title'].values, columns=movies_df['Title'].values)


# ---------------------------------------------------------
# 6. Recommendation functions
# ---------------------------------------------------------
def user_based_cf(user_id, user_item_matrix, user_sim, k=10, top_n=10):
    if user_id not in user_item_matrix.index:
        return []
    neighbors = user_sim.loc[user_id].sort_values(ascending=False).drop(labels=[user_id], errors='ignore').head(k).index
    target = user_item_matrix.loc[user_id]
    unrated = target[target == 0].index
    recs = []
    for movie in unrated:
        num, den = 0.0, 0.0
        for n in neighbors:
            r = user_item_matrix.loc[n, movie]
            if r > 0:
                w = user_sim.loc[user_id, n]
                num += w * r
                den += abs(w)
        if den > 0:
            recs.append((movie, num / den))
    return sorted(recs, key=lambda x: x[1], reverse=True)[:top_n]

def item_based_cf(user_id, user_item_matrix, item_sim, top_n=10):
    if user_id not in user_item_matrix.index:
        return []
    user_ratings = user_item_matrix.loc[user_id]
    rated = user_ratings[user_ratings > 0].index
    unrated = user_ratings[user_ratings == 0].index
    recs = []
    for movie in unrated:
        num, den = 0.0, 0.0
        for r in rated:
            s = item_sim.loc[movie, r]
            num += s * user_ratings[r]
            den += abs(s)
        if den > 0:
            recs.append((movie, num / den))
    return sorted(recs, key=lambda x: x[1], reverse=True)[:top_n]

def content_based_recs(movie_title, content_sim, top_n=10):
    if movie_title not in content_sim.index:
        return []
    sims = content_sim[movie_title].sort_values(ascending=False).drop(labels=[movie_title], errors='ignore')
    return list(zip(sims.index, sims.values))[:top_n]

def movie_to_movie_hybrid(movie_title, movies_df, content_sim, w_content=0.75, w_numeric=0.25, top_n=10):
    if movie_title not in content_sim.index:
        return pd.DataFrame(columns=['Title','Score'])
    numeric_sim = compute_numeric_similarity(movies_df)
    hybrid_sim = (w_content * content_sim.values) + (w_numeric * numeric_sim.values)
    hybrid_df = pd.DataFrame(hybrid_sim, index=movies_df['Title'].values, columns=movies_df['Title'].values)
    sims = hybrid_df[movie_title].sort_values(ascending=False).drop(labels=[movie_title], errors='ignore')
    top = sims.head(top_n).index
    out = movies_df[movies_df['Title'].isin(top)][['Title','Genre','Rating','Director','Cast']].copy()
    out['score'] = out['Title'].map(lambda t: sims.loc[t])
    return out.sort_values('score', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # Load movies
    movies = load_movies("imdb-movies-dataset.csv")
    print(f"Loaded {len(movies)} movies.")

    # Create synthetic users
    n_synth_users = 120
    print(f"Generating {n_synth_users} synthetic users...")
    ratings = generate_synthetic_ratings(movies, n_users=n_synth_users)

    uim = build_user_item_matrix(ratings, movies['Title'].values)
    print("User-item matrix:", uim.shape)

    user_sim = pd.DataFrame(cosine_similarity(uim), index=uim.index, columns=uim.index)
    item_sim = item_similarity_from_user_item(uim)

    # SVD
    print("\nComputing SVD predictions...")
    svd_pred, _ = compute_svd_predictions(uim, n_components=20)
    print(svd_pred.head())

    # NMF
    print("\nComputing NMF predictions...")
    try:
        nmf_pred, _ = compute_nmf_predictions(uim, n_components=20)
        print(nmf_pred.head())
    except Exception as e:
        print("NMF failed:", e)

    # Demo CF
    example_user = uim.index[0]
    ubcf = user_based_cf(example_user, uim, user_sim, top_n=5)
    ubcf_df = pd.DataFrame(ubcf, columns=['Title', 'Score'])
    ubcf_df = ubcf_df.merge(movies[['Title', 'Poster']], on='Title', how='left')
    print("\nUser-based CF top 5 with Posters:")
    print(ubcf_df)

    ibcf = item_based_cf(example_user, uim, item_sim, top_n=5)
    ibcf_df = pd.DataFrame(ibcf, columns=['Title', 'Score'])
    ibcf_df = ibcf_df.merge(movies[['Title', 'Poster']], on='Title', how='left')
    print("\nItem-based CF top 5 with Posters:")
    print(ibcf_df)


    # -----------------------------------------------------
    # USER INPUT FOR CONTENT + HYBRID RECOMMENDATIONS
    # -----------------------------------------------------
    print("\nComputing content similarity...")
    content_sim = compute_content_similarity(movies, max_features=8000)

    # Input from user
    user_movie = input("\nEnter a movie name: ").strip()

    if user_movie not in movies['Title'].values:
        print(f"\nMovie '{user_movie}' not found!")

        suggestions = [t for t in movies['Title'].values if user_movie.lower() in t.lower()]
        if suggestions:
            print("\nDid you mean:")
            for s in suggestions[:5]:
                print(" -", s)
        else:
            print("No similar movies found.")
    else:
        print(f"\nContent-based recommendations for '{user_movie}':")
        cb = content_based_recs(user_movie, content_sim, top_n=10)
        cb_df = pd.DataFrame(cb, columns=['Title', 'Similarity'])
        cb_df = cb_df.merge(movies[['Title', 'Poster']], on='Title', how='left')
        print(cb_df)


        print(f"\nHybrid recommendations for '{user_movie}':")
        hybrid = movie_to_movie_hybrid(user_movie, movies, content_sim, top_n=10)
        hybrid = hybrid.merge(movies[['Title', 'Poster']], on='Title', how='left')
        print(hybrid)


    print("\nDone.")
#--------- add Poster column from the dataset to every model result.
