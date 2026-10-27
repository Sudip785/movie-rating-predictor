"""
Movie Rating Predictor - Inference Module
=========================================
This module provides a standalone prediction function and CLI tool using the saved,
genuinely trained Linear Regression model and preprocessing pipeline.

Ensures strict parity with the training pipeline:
- Same release year parsing and training-median imputation
- Identical multi-hot genre encoding
- Identical historical training statistics lookup with Bayesian smoothing
- Feature ordering matching training feature_cols
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import joblib

# Ensure safe console output encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_default_paths() -> Tuple[Path, Path, Path]:
    """
    Resolve paths to project root, data directory, and models directory.
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    return project_root, data_dir, models_dir


class MovieRatingPredictor:
    """
    Encapsulates trained model artifacts and identical preprocessing for rating inference.
    """

    def __init__(self, models_dir: Optional[Path] = None, data_dir: Optional[Path] = None):
        _, default_data_dir, default_models_dir = get_default_paths()
        self.models_dir = Path(models_dir) if models_dir else default_models_dir
        self.data_dir = Path(data_dir) if data_dir else default_data_dir

        self.lr_model = None
        self.dummy_model = None
        self.artifacts = None
        self.movies_df = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Load model and preprocessing artifacts from disk.
        """
        lr_path = self.models_dir / "linear_regression_model.joblib"
        dummy_path = self.models_dir / "dummy_baseline_model.joblib"
        artifacts_path = self.models_dir / "preprocessing_artifacts.joblib"

        if not (lr_path.exists() and dummy_path.exists() and artifacts_path.exists()):
            raise FileNotFoundError(
                f"Trained model artifacts were not found in: {self.models_dir}\n"
                f"Please train the model first by running:\n"
                f"    python src/train_model.py\n"
            )

        self.lr_model = joblib.load(lr_path)
        self.dummy_model = joblib.load(dummy_path)
        self.artifacts = joblib.load(artifacts_path)

        # Preprocessing state saved during training
        self.feature_cols: List[str] = self.artifacts["feature_cols"]
        self.standard_genres: List[str] = self.artifacts["standard_genres"]
        self.median_year: float = self.artifacts["median_year"]
        self.global_mean: float = self.artifacts["global_mean"]
        self.m_prior: float = self.artifacts["m_prior"]
        self.movie_stats_dict: Dict[int, Dict[str, Any]] = self.artifacts["movie_stats_dict"]

        # Optionally load movies.csv for title lookup
        movies_csv = self.data_dir / "movies.csv"
        if movies_csv.exists():
            self.movies_df = pd.read_csv(movies_csv)
        else:
            self.movies_df = None

    def _extract_year(self, title: str) -> float:
        """
        Extract release year from movie title with fallback to training median.
        """
        match = re.search(r'\((\d{4})\)', str(title))
        if match:
            year = int(match.group(1))
            if 1880 <= year <= 2030:
                return float(year)
        return self.median_year

    def build_feature_vector(
        self,
        title: str,
        genres: str,
        movie_id: Optional[int] = None,
        custom_year: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Transform raw movie metadata into the exact feature vector expected by the model.
        """
        # 1. Release year
        if custom_year is not None:
            release_year = float(custom_year)
        else:
            release_year = self._extract_year(title)

        # 2. Genre parsing
        if isinstance(genres, list):
            genre_tokens = [g.strip() for g in genres]
        elif isinstance(genres, str):
            genre_tokens = [g.strip() for g in genres.split("|") if g.strip()]
        else:
            genre_tokens = []

        is_no_genre = (len(genre_tokens) == 0) or ("(no genres listed)" in genre_tokens)
        genre_count = float(len(genre_tokens)) if not is_no_genre else 0.0

        # 3. Multi-hot genre indicators
        genre_indicators = {}
        for genre in self.standard_genres:
            genre_indicators[f"genre_{genre}"] = 1.0 if genre in genre_tokens else 0.0

        # 4. Historical rating statistics from training set
        if movie_id is not None and movie_id in self.movie_stats_dict:
            stats = self.movie_stats_dict[movie_id]
            train_count = stats["count"]
            train_sum = stats["sum"]
            is_in_train = True
        else:
            train_count = 0
            train_sum = 0.0
            is_in_train = False

        hist_movie_avg_rating = (train_sum + self.m_prior * self.global_mean) / (
            train_count + self.m_prior
        )
        log_rating_count = float(np.log1p(train_count))

        # 5. Assemble dictionary matching training feature order
        feature_dict = {
            "release_year": release_year,
            "genre_count": genre_count,
            **genre_indicators,
            "hist_movie_avg_rating": hist_movie_avg_rating,
            "log_rating_count": log_rating_count
        }

        # Build single-row DataFrame with exact columns
        features_df = pd.DataFrame([feature_dict])[self.feature_cols]

        metadata = {
            "title": title,
            "movie_id": movie_id,
            "genres": genre_tokens,
            "release_year": int(release_year),
            "is_in_training_data": is_in_train,
            "training_review_count": train_count,
            "historical_mean_raw": (train_sum / train_count) if train_count > 0 else None,
            "historical_mean_smoothed": hist_movie_avg_rating
        }

        return features_df, metadata

    def predict(
        self,
        title: str,
        genres: str,
        movie_id: Optional[int] = None,
        custom_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a movie rating prediction along with full interpretability details.
        """
        features_df, metadata = self.build_feature_vector(
            title=title,
            genres=genres,
            movie_id=movie_id,
            custom_year=custom_year
        )

        # 1. Model inference
        raw_pred = float(self.lr_model.predict(features_df)[0])
        dummy_pred = float(self.dummy_model.predict(features_df)[0])

        # 2. Clip prediction to valid MovieLens scale [0.5, 5.0]
        clipped_pred = max(0.5, min(5.0, raw_pred))

        # 3. Calculate feature contributions: beta_i * x_i
        intercept = float(self.lr_model.intercept_)
        contributions = {}
        for col, coef in zip(self.feature_cols, self.lr_model.coef_):
            val = float(features_df[col].iloc[0])
            contrib = float(coef * val)
            contributions[col] = {
                "value": val,
                "coefficient": float(coef),
                "contribution": contrib
            }

        return {
            "title": metadata["title"],
            "movie_id": metadata["movie_id"],
            "release_year": metadata["release_year"],
            "genres": metadata["genres"],
            "is_in_training_data": metadata["is_in_training_data"],
            "training_review_count": metadata["training_review_count"],
            "historical_mean_smoothed": metadata["historical_mean_smoothed"],
            "predicted_rating": round(clipped_pred, 2),
            "predicted_rating_raw": round(raw_pred, 4),
            "dummy_baseline_prediction": round(dummy_pred, 2),
            "intercept": round(intercept, 4),
            "features_used": features_df.to_dict(orient="records")[0],
            "feature_contributions": contributions
        }

    def predict_by_title(self, search_title: str) -> Dict[str, Any]:
        """
        Look up a movie in the dataset by title (case-insensitive substring) and predict.
        """
        if self.movies_df is None:
            raise RuntimeError("movies.csv is required for title lookup.")

        matches = self.movies_df[
            self.movies_df["title"].str.contains(search_title, case=False, na=False, regex=False)
        ]

        if matches.empty:
            raise ValueError(f"No movie found matching '{search_title}'.")

        row = matches.iloc[0]
        return self.predict(
            title=row["title"],
            genres=row["genres"],
            movie_id=int(row["movieId"])
        )

    def predict_by_movie_id(self, movie_id: int) -> Dict[str, Any]:
        """
        Look up a movie by exact movieId and predict.
        """
        if self.movies_df is None:
            raise RuntimeError("movies.csv is required for movieId lookup.")

        matches = self.movies_df[self.movies_df["movieId"] == movie_id]
        if matches.empty:
            raise ValueError(f"Movie with movieId={movie_id} not found in movies.csv.")

        row = matches.iloc[0]
        return self.predict(
            title=row["title"],
            genres=row["genres"],
            movie_id=int(row["movieId"])
        )


def main() -> None:
    """
    CLI interface for running predictions from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Predict movie rating using the trained Linear Regression model."
    )
    parser.add_argument(
        "title",
        nargs="?",
        default="Toy Story (1995)",
        help="Movie title to look up or predict (default: 'Toy Story (1995)')"
    )
    parser.add_argument(
        "--movieId",
        type=int,
        default=None,
        help="Exact movieId from MovieLens dataset"
    )
    parser.add_argument(
        "--genres",
        type=str,
        default=None,
        help="Pipe-separated genres for custom movie (e.g. 'Action|Adventure|Sci-Fi')"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Release year for custom movie"
    )

    args = parser.parse_args()

    try:
        predictor = MovieRatingPredictor()

        if args.movieId is not None:
            result = predictor.predict_by_movie_id(args.movieId)
        elif args.genres is not None:
            result = predictor.predict(
                title=args.title,
                genres=args.genres,
                custom_year=args.year
            )
        else:
            result = predictor.predict_by_title(args.title)

        print("\n" + "=" * 65)
        print("PREDICTED MOVIE RATING")
        print("=" * 65)
        print(f"Title                 : {result['title']}")
        if result['movie_id']:
            print(f"Movie ID              : {result['movie_id']}")
        print(f"Release Year          : {result['release_year']}")
        print(f"Genres                : {', '.join(result['genres']) if result['genres'] else 'None'}")
        print(f"In Training Data?     : {'Yes' if result['is_in_training_data'] else 'No (Cold-start / New)'}")
        print(f"Training Reviews      : {result['training_review_count']}")
        print("-" * 65)
        print(f"★ PREDICTED RATING    : {result['predicted_rating']} / 5.0  (raw: {result['predicted_rating_raw']})")
        print(f"Dummy Baseline Rating : {result['dummy_baseline_prediction']} / 5.0  (always predicts mean)")
        print("-" * 65)

        print("\nTop Contributing Features:")
        contribs = sorted(
            result["feature_contributions"].items(),
            key=lambda x: abs(x[1]["contribution"]),
            reverse=True
        )
        for name, item in contribs[:6]:
            if abs(item["contribution"]) > 0.0001:
                print(f"  • {name:<24}: val={item['value']:<5.1f} | coef={item['coefficient']:<+7.4f} -> effect={item['contribution']:+.3f}")

        print("\n[NOTE] Prediction represents the general expected audience rating")
        print("based on genre composition, release era, and historical crowd reception.\n")

    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
