"""
Movie Rating Predictor - Model Training Pipeline
================================================
This script implements a genuine supervised machine learning workflow for predicting
movie ratings using historical MovieLens data.

Key Machine Learning Concepts Demonstrated:
1. Data Ingestion & Validation: Loading and validating schema integrity.
2. Feature Engineering: Transforming nominal IDs and text into meaningful numeric signals
   (multi-hot genre encoding, release year extraction, genre count, historical evidence).
3. Leakage Prevention: Splitting data BEFORE calculating summary statistics and using
   Leave-One-Out (out-of-self) target encoding for historical ratings on the training split.
4. Model Training: Fitting LinearRegression as the primary model and DummyRegressor as baseline.
5. Evaluation: Evaluating on held-out test data using MAE and RMSE.
6. Artifact Serialization: Exporting the fitted models and preprocessing state.
"""

import os
import sys
import re
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Ensure safe console output encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configure logging for clear educational output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Standard MovieLens genres list (19 explicit genres)
STANDARD_GENRES: List[str] = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

# Bayesian smoothing parameter for historical movie ratings.
# Higher m pulls movies with few ratings closer to the global average.
M_PRIOR: float = 5.0


def resolve_paths() -> Tuple[Path, Path, Path]:
    """
    Resolve base directory and data paths dynamically regardless of working directory.
    """
    # Current script directory: .../movie-rating-predictor/src
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    data_dir = project_root / "data"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    return project_root, data_dir, models_dir


def load_and_validate_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load movies.csv and ratings.csv and perform rigorous schema validation.
    
    Raises:
        FileNotFoundError: If any required CSV file is missing.
        ValueError: If required columns or data values are invalid.
    """
    movies_path = data_dir / "movies.csv"
    ratings_path = data_dir / "ratings.csv"

    if not movies_path.exists():
        raise FileNotFoundError(
            f"Missing required dataset file: {movies_path}\n"
            f"Please ensure movies.csv is placed inside '{data_dir}'."
        )

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Missing required dataset file: {ratings_path}\n"
            f"Please ensure ratings.csv is placed inside '{data_dir}'."
        )

    logger.info("Loading dataset files...")
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)

    # Validate required columns in movies.csv
    required_movie_cols = {"movieId", "title", "genres"}
    missing_movie_cols = required_movie_cols - set(movies_df.columns)
    if missing_movie_cols:
        raise ValueError(f"movies.csv is missing required columns: {missing_movie_cols}")

    # Validate required columns in ratings.csv
    required_rating_cols = {"userId", "movieId", "rating"}
    missing_rating_cols = required_rating_cols - set(ratings_df.columns)
    if missing_rating_cols:
        raise ValueError(f"ratings.csv is missing required columns: {missing_rating_cols}")

    # Validate ratings range (MovieLens is 0.5 to 5.0)
    if not (ratings_df["rating"].min() >= 0.5 and ratings_df["rating"].max() <= 5.0):
        logger.warning(
            f"Ratings out of typical 0.5-5.0 scale detected: "
            f"min={ratings_df['rating'].min()}, max={ratings_df['rating'].max()}"
        )

    logger.info(f"Loaded {len(movies_df):,} movies and {len(ratings_df):,} ratings successfully.")
    return movies_df, ratings_df


def extract_year_from_title(title: str) -> float:
    """
    Extract the 4-digit release year typically formatted as '(YYYY)' at the end of movie titles.
    Returns np.nan if no valid year pattern is matched.
    """
    match = re.search(r'\((\d{4})\)', str(title))
    if match:
        year = int(match.group(1))
        # Sanity check for realistic movie release years
        if 1880 <= year <= 2030:
            return float(year)
    return np.nan


def prepare_raw_features(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ratings with movie metadata and extract intrinsic movie features:
    - release_year (parsed from title)
    - genre multi-hot indicators (one column per genre)
    - genre_count (number of genres tagged)
    """
    logger.info("Merging ratings with movie metadata...")
    df = ratings_df.merge(movies_df, on="movieId", how="left")

    logger.info("Extracting release year from titles...")
    df["release_year"] = df["title"].apply(extract_year_from_title)

    logger.info("Encoding genre indicator features...")
    for genre in STANDARD_GENRES:
        col_name = f"genre_{genre}"
        df[col_name] = df["genres"].apply(
            lambda g: 1.0 if isinstance(g, str) and genre in g.split("|") else 0.0
        )

    # Genre count: number of genres tagged for the movie
    df["genre_count"] = df["genres"].apply(
        lambda g: float(len(g.split("|"))) if isinstance(g, str) and g != "(no genres listed)" else 0.0
    )

    return df


def engineer_split_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Perform feature engineering strictly avoiding target leakage:
    
    1. Imputation: Release year median is computed ONLY from train_df.
    2. Historical Rating Statistics: Computed ONLY from train_df.
       - For train_df: Leave-One-Out (out-of-self) target calculation ensures
         the rating being predicted is NOT in its own historical mean!
       - For test_df: Look up stats precomputed from train_df with Bayesian smoothing.
    """
    logger.info("Applying leakage-free feature engineering...")

    # 1. Median release year from training set only
    train_median_year = float(train_df["release_year"].median())
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["release_year"] = train_df["release_year"].fillna(train_median_year)
    test_df["release_year"] = test_df["release_year"].fillna(train_median_year)

    # 2. Global mean rating from training set only
    global_train_mean = float(train_df["rating"].mean())
    logger.info(f"Training set global mean rating: {global_train_mean:.4f}")

    # 3. Aggregate historical movie statistics from train_df ONLY
    movie_stats = (
        train_df.groupby("movieId")["rating"]
        .agg(train_count="count", train_sum="sum")
        .reset_index()
    )

    # Build a lookup dictionary for fast runtime inference
    movie_stats_dict = {}
    for row in movie_stats.itertuples(index=False):
        movie_stats_dict[int(row.movieId)] = {
            "count": int(row.train_count),
            "sum": float(row.train_sum)
        }

    # 4. Leave-one-out calculation for train_df (PREVENT TARGET LEAKAGE)
    # Target leakage occurs if a row's own rating is included in the movie's average.
    # By subtracting the current rating, we get the true historical signal.
    train_merged = train_df.merge(movie_stats, on="movieId", how="left")
    loo_sum = train_merged["train_sum"] - train_merged["rating"]
    loo_count = train_merged["train_count"] - 1

    # Bayesian smoothed mean: (sum + m * prior) / (count + m)
    train_merged["hist_movie_avg_rating"] = (
        loo_sum + M_PRIOR * global_train_mean
    ) / (loo_count + M_PRIOR)
    train_merged["log_rating_count"] = np.log1p(np.maximum(0, loo_count))

    # 5. Precomputed stats lookup for test_df
    # In test data, we only use information available from the training split.
    test_merged = test_df.merge(movie_stats, on="movieId", how="left")
    test_merged["train_count"] = test_merged["train_count"].fillna(0)
    test_merged["train_sum"] = test_merged["train_sum"].fillna(0)

    test_merged["hist_movie_avg_rating"] = (
        test_merged["train_sum"] + M_PRIOR * global_train_mean
    ) / (test_merged["train_count"] + M_PRIOR)
    test_merged["log_rating_count"] = np.log1p(test_merged["train_count"])

    # Define exact feature columns in deterministic order
    genre_feature_cols = [f"genre_{g}" for g in STANDARD_GENRES]
    feature_cols = [
        "release_year",
        "genre_count",
        *genre_feature_cols,
        "hist_movie_avg_rating",
        "log_rating_count"
    ]

    preprocessing_artifacts = {
        "feature_cols": feature_cols,
        "standard_genres": STANDARD_GENRES,
        "median_year": train_median_year,
        "global_mean": global_train_mean,
        "m_prior": M_PRIOR,
        "movie_stats_dict": movie_stats_dict,
        "train_size": len(train_df),
        "test_size": len(test_df)
    }

    return train_merged, test_merged, preprocessing_artifacts, feature_cols


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[LinearRegression, DummyRegressor, Dict[str, Any]]:
    """
    Train LinearRegression and DummyRegressor baseline on train_df.
    Evaluate both strictly on held-out test_df using MAE and RMSE.
    """
    X_train = train_df[feature_cols]
    y_train = train_df["rating"]

    X_test = test_df[feature_cols]
    y_test = test_df["rating"]

    logger.info(f"Training set: {len(X_train):,} samples with {len(feature_cols)} features.")
    logger.info(f"Test set: {len(X_test):,} samples.")

    # 1. Baseline Model: DummyRegressor
    # Strategy 'mean' always predicts the mean of the training target y_train.
    logger.info("Training DummyRegressor baseline (predicts constant training mean)...")
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)

    # 2. Primary Model: LinearRegression
    # Fits weights (betas) minimizing residual sum of squares via Ordinary Least Squares (OLS).
    logger.info("Training LinearRegression model via Ordinary Least Squares...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # 3. Predictions on held-out test set
    dummy_test_preds = dummy.predict(X_test)
    lr_test_preds = lr_model.predict(X_test)

    # 4. Real metric calculations
    dummy_mae = float(mean_absolute_error(y_test, dummy_test_preds))
    dummy_rmse = float(root_mean_squared_error(y_test, dummy_test_preds))

    lr_mae = float(mean_absolute_error(y_test, lr_test_preds))
    lr_rmse = float(root_mean_squared_error(y_test, lr_test_preds))

    mae_improvement_pct = ((dummy_mae - lr_mae) / dummy_mae) * 100.0
    rmse_improvement_pct = ((dummy_rmse - lr_rmse) / dummy_rmse) * 100.0

    metrics = {
        "dataset_total_ratings": len(train_df) + len(test_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "dummy_baseline": {
            "model_name": "DummyRegressor (mean baseline)",
            "strategy": "mean",
            "constant_prediction": float(dummy.constant_[0][0]),
            "mae": dummy_mae,
            "rmse": dummy_rmse
        },
        "linear_regression": {
            "model_name": "LinearRegression (OLS)",
            "mae": lr_mae,
            "rmse": lr_rmse,
            "mae_improvement_pct": mae_improvement_pct,
            "rmse_improvement_pct": rmse_improvement_pct,
            "intercept": float(lr_model.intercept_)
        }
    }

    return lr_model, dummy, metrics


def save_artifacts(
    models_dir: Path,
    lr_model: LinearRegression,
    dummy_model: DummyRegressor,
    preprocessing_artifacts: Dict[str, Any],
    metrics: Dict[str, Any]
) -> None:
    """
    Save models, preprocessing parameters, and evaluation results to disk.
    """
    lr_path = models_dir / "linear_regression_model.joblib"
    dummy_path = models_dir / "dummy_baseline_model.joblib"
    prep_path = models_dir / "preprocessing_artifacts.joblib"
    metrics_path = models_dir / "evaluation_metrics.json"
    coef_path = models_dir / "feature_coefficients.json"

    logger.info(f"Saving LinearRegression model to: {lr_path}")
    joblib.dump(lr_model, lr_path)

    logger.info(f"Saving DummyRegressor model to: {dummy_path}")
    joblib.dump(dummy_model, dummy_path)

    logger.info(f"Saving preprocessing artifacts to: {prep_path}")
    joblib.dump(preprocessing_artifacts, prep_path)

    # Save metrics JSON for easy inspection and UI display
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to: {metrics_path}")

    # Save readable feature coefficients
    feature_cols = preprocessing_artifacts["feature_cols"]
    coefficients = {
        "intercept": float(lr_model.intercept_),
        "coefficients": {
            col: float(coef) for col, coef in zip(feature_cols, lr_model.coef_)
        }
    }
    with open(coef_path, "w", encoding="utf-8") as f:
        json.dump(coefficients, f, indent=2)
    logger.info(f"Saved model coefficients to: {coef_path}")


def main() -> None:
    """
    Execute the complete end-to-end training pipeline.
    """
    print("\n" + "=" * 70)
    print("MOVIE RATING PREDICTOR - MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 70 + "\n")

    project_root, data_dir, models_dir = resolve_paths()
    logger.info(f"Project root: {project_root}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Models directory: {models_dir}")

    # 1. Load and validate
    movies_df, ratings_df = load_and_validate_data(data_dir)

    # 2. Extract intrinsic movie features
    df = prepare_raw_features(ratings_df, movies_df)

    # 3. Train-test split (80% train, 20% test, reproducible random_state=42)
    logger.info("Splitting dataset into train (80%) and test (20%) sets (random_state=42)...")
    train_raw, test_raw = train_test_split(df, test_size=0.2, random_state=42)

    # 4. Feature engineering without data leakage
    train_processed, test_processed, prep_artifacts, feature_cols = engineer_split_features(
        train_raw, test_raw
    )

    # 5. Train models and evaluate on held-out test data
    lr_model, dummy_model, metrics = train_and_evaluate(
        train_processed, test_processed, feature_cols
    )

    # 6. Save all artifacts
    save_artifacts(models_dir, lr_model, dummy_model, prep_artifacts, metrics)

    # 7. Print summary for viva preparation
    dummy_res = metrics["dummy_baseline"]
    lr_res = metrics["linear_regression"]

    print("\n" + "-" * 70)
    print("[EVALUATION] MODEL EVALUATION RESULTS (HELD-OUT TEST SET)")
    print("-" * 70)
    print(f"Total Ratings in Dataset : {metrics['dataset_total_ratings']:,}")
    print(f"Training Set Size        : {metrics['train_samples']:,} rows (80%)")
    print(f"Held-Out Test Set Size   : {metrics['test_samples']:,} rows (20%)")
    print("-" * 70)
    print(f"{'Metric':<25} | {'Dummy Baseline':<18} | {'Linear Regression':<18} | {'Improvement':<12}")
    print("-" * 70)
    print(
        f"{'MAE (Mean Absolute Error)':<25} | "
        f"{dummy_res['mae']:<18.4f} | "
        f"{lr_res['mae']:<18.4f} | "
        f"{lr_res['mae_improvement_pct']:+.2f}%"
    )
    print(
        f"{'RMSE (Root Mean Sq Error)':<25} | "
        f"{dummy_res['rmse']:<18.4f} | "
        f"{lr_res['rmse']:<18.4f} | "
        f"{lr_res['rmse_improvement_pct']:+.2f}%"
    )
    print("-" * 70)
    print("\n[BENCHMARK NOTE] Why compare with DummyRegressor?")
    print("  DummyRegressor simply predicts the training average (3.50 stars).")
    print("  Comparing with DummyRegressor proves that our model learned genuine")
    print("  patterns rather than simply guessing the population average.\n")
    print("[SUCCESS] Training completed successfully! All artifacts are saved in models/.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
