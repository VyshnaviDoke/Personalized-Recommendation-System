from flask import Flask, render_template, request
import pandas as pd
from recommender import (
    load_movies, compute_content_similarity, movie_to_movie_hybrid,
    content_based_recs, user_based_cf, item_based_cf,
    build_user_item_matrix, generate_synthetic_ratings,
    compute_svd_predictions, compute_nmf_predictions
)
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------------------------------------------------
# Load Movies + Build User Models One Time
# ---------------------------------------------------
movies = load_movies("imdb-movies-dataset.csv")

ratings = generate_synthetic_ratings(movies, n_users=120)
uim = build_user_item_matrix(ratings, movies["Title"].values)

user_sim = pd.DataFrame(cosine_similarity(uim), index=uim.index, columns=uim.index)
item_sim = pd.DataFrame(cosine_similarity(uim.T), index=uim.columns, columns=uim.columns)

content_sim = compute_content_similarity(movies)

svd_pred, _ = compute_svd_predictions(uim, n_components=20)
nmf_pred, _ = compute_nmf_predictions(uim, n_components=20)


def top_n(pred_matrix, user_id, movies, n=10):
    row = pred_matrix.loc[user_id]
    top = row.sort_values(ascending=False).head(n)
    df = pd.DataFrame({"Title": top.index, "Predicted Rating": top.values})
    return df.merge(movies[["Title", "Poster"]], on="Title", how="left")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/results", methods=["POST"])
def results():
    user_input = request.form.get("user_id").strip()
    movie_input = request.form.get("movie_name").strip()

    # -------------------------------------------
    # OUTPUT CONTAINERS
    # -------------------------------------------
    ubcf = ibcf = svd = nmf = None
    cb = hybrid = None
    error_user = error_movie = None

    # ===========================================
    # 1️⃣ USER-BASED OUTPUTS
    # ===========================================
    if user_input != "":
        try:
            user_id = int(user_input)

            if user_id not in uim.index:
                error_user = f"User ID {user_id} not found! Valid range: {uim.index.min()} - {uim.index.max()}"
            else:
                # User-Based CF
                ub_raw = user_based_cf(user_id, uim, user_sim, top_n=10)
                ubcf = pd.DataFrame(ub_raw, columns=["Title", "Score"]).merge(
                    movies[["Title", "Poster"]], on="Title", how="left"
                ).to_dict(orient="records")

                # Item-Based CF
                ib_raw = item_based_cf(user_id, uim, item_sim, top_n=10)
                ibcf = pd.DataFrame(ib_raw, columns=["Title", "Score"]).merge(
                    movies[["Title", "Poster"]], on="Title", how="left"
                ).to_dict(orient="records")

                # SVD
                svd = top_n(svd_pred, user_id, movies).to_dict(orient="records")

                # NMF
                nmf = top_n(nmf_pred, user_id, movies).to_dict(orient="records")

        except:
            error_user = "Invalid user ID (must be a number)."


    # ===========================================
    # 2️⃣ MOVIE-BASED OUTPUTS
    # ===========================================
    if movie_input != "":
        if movie_input not in movies["Title"].values:
            error_movie = f"Movie '{movie_input}' not found!"

            suggestions = [t for t in movies["Title"] if movie_input.lower() in t.lower()]

            return render_template(
                "results.html",
                user_input=user_input,
                movie_input=movie_input,
                error_user=error_user,
                error_movie=error_movie,
                suggestions=suggestions[:5],
                ubcf=ubcf, ibcf=ibcf, svd=svd, nmf=nmf,
                cb=None, hybrid=None
            )
        else:
            # Content-based
            cb_raw = content_based_recs(movie_input, content_sim, top_n=10)
            cb = pd.DataFrame(cb_raw, columns=["Title", "Similarity"]).merge(
                movies[["Title", "Poster"]], on="Title", how="left"
            ).to_dict(orient="records")

            # Hybrid
            hybrid_df = movie_to_movie_hybrid(movie_input, movies, content_sim, top_n=10)
            hybrid = hybrid_df.merge(
                movies[["Title", "Poster"]], on="Title", how="left"
            ).to_dict(orient="records")

    return render_template(
        "results.html",
        user_input=user_input,
        movie_input=movie_input,
        error_user=error_user,
        error_movie=error_movie,
        ubcf=ubcf,
        ibcf=ibcf,
        svd=svd,
        nmf=nmf,
        cb=cb,
        hybrid=hybrid
    )


if __name__ == "__main__":
    app.run(debug=True)
