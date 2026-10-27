"""
Movie Rating Predictor - Web Application
========================================
A clean, focused movie rating prediction tool powered by machine learning.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np

# Ensure src/ is in Python path for importing inference module
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from predict import MovieRatingPredictor, get_default_paths
except ImportError as e:
    st.error(f"Failed to import prediction module: {e}")
    st.stop()

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="Movie Rating Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, professional styling
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a237e;
        margin-bottom: 0.1rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #546e7a;
        margin-bottom: 1.8rem;
    }
    .rating-card {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        color: white;
        border-radius: 14px;
        padding: 28px 24px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(26, 35, 126, 0.2);
        margin-bottom: 20px;
    }
    .rating-score {
        font-size: 3.6rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .rating-label {
        font-size: 1.05rem;
        opacity: 0.9;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .verdict-tag {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 10px;
        letter-spacing: 0.3px;
    }
    .verdict-masterpiece {
        background-color: #ffd700;
        color: #1a1a1a;
    }
    .verdict-great {
        background-color: #00e676;
        color: #003300;
    }
    .verdict-good {
        background-color: #64b5f6;
        color: #0d47a1;
    }
    .verdict-mixed {
        background-color: #ffb74d;
        color: #e65100;
    }
    .verdict-low {
        background-color: #ff8a80;
        color: #b71c1c;
    }
    .badge-genre {
        display: inline-block;
        background-color: #e8eaf6;
        color: #1a237e;
        padding: 5px 12px;
        border-radius: 14px;
        font-size: 0.85rem;
        margin-right: 6px;
        margin-bottom: 6px;
        font-weight: 600;
        border: 1px solid #c5cae9;
    }
    .movie-info-box {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """
    Cache the MovieRatingPredictor instance to avoid reloading models on every interaction.
    """
    project_root, data_dir, models_dir = get_default_paths()
    return MovieRatingPredictor(models_dir=models_dir, data_dir=data_dir)


def get_verdict_info(rating: float) -> Tuple[str, str]:
    """
    Return human-readable verdict and CSS class for rating.
    """
    if rating >= 4.2:
        return "Must Watch / Critical Masterpiece 🏆", "verdict-masterpiece"
    elif rating >= 3.7:
        return "Highly Recommended / Strong Reception ⭐", "verdict-great"
    elif rating >= 3.2:
        return "Decent Watch / Positive Audience Score 🍿", "verdict-good"
    elif rating >= 2.8:
        return "Mixed Reactions / Polarizing ⚖️", "verdict-mixed"
    else:
        return "Below Average / Niche ⚠️", "verdict-low"


def main():
    # Header
    st.markdown('<div class="hero-title">🎬 Movie Rating Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Predict audience ratings for any movie based on its metadata and genre profile.</div>',
        unsafe_allow_html=True
    )

    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(
            "⚠️ **Model Artifacts Not Found!** "
            f"Please run `python src/train_model.py` first. Error: {e}"
        )
        return

    # Two column layout: Left for inputs, Right for predictions
    col_input, col_result = st.columns([1.1, 1], gap="large")

    pred_result = None

    with col_input:
        st.subheader("Movie Information")

        input_mode = st.radio(
            "Select Input Type:",
            ["Choose from Catalog (9,700+ Movies)", "Enter a Custom / New Movie"],
            horizontal=True
        )

        if input_mode == "Choose from Catalog (9,700+ Movies)":
            if predictor.movies_df is not None and not predictor.movies_df.empty:
                movie_titles = predictor.movies_df["title"].tolist()
                
                default_idx = 0
                if "Toy Story (1995)" in movie_titles:
                    default_idx = movie_titles.index("Toy Story (1995)")

                selected_title = st.selectbox(
                    "Search or Select Movie Title:",
                    options=movie_titles,
                    index=default_idx
                )

                # Show preview of genres for selected movie
                matched = predictor.movies_df[predictor.movies_df["title"] == selected_title]
                if not matched.empty:
                    row_genres = [g for g in str(matched.iloc[0]["genres"]).split("|") if g]
                    badges_preview = "".join([f"<span class='badge-genre'>{g}</span>" for g in row_genres])
                    st.markdown(f"<div style='margin-top: 6px; margin-bottom: 14px;'>{badges_preview}</div>", unsafe_allow_html=True)

                if st.button("Predict Rating", type="primary", use_container_width=True):
                    try:
                        pred_result = predictor.predict_by_title(selected_title)
                    except Exception as err:
                        st.error(f"Error predicting rating: {err}")

            else:
                st.warning("Movie catalog is unavailable.")

        else:
            # Custom movie inputs
            custom_title = st.text_input("Movie Title", value="Project Odyssey", placeholder="e.g. Inception 2")
            custom_year = st.number_input("Release Year", min_value=1920, max_value=2030, value=2025, step=1)

            st.write("Genres (Choose all that apply):")
            genre_cols = st.columns(3)
            selected_genres = []
            
            for idx, genre in enumerate(predictor.standard_genres):
                if genre == "(no genres listed)":
                    continue
                with genre_cols[idx % 3]:
                    is_checked = genre in ["Action", "Sci-Fi"]
                    if st.checkbox(genre, value=is_checked, key=f"genre_box_{genre}"):
                        selected_genres.append(genre)

            if st.button("Predict Rating", type="primary", use_container_width=True):
                if not custom_title.strip():
                    st.warning("Please enter a movie title.")
                else:
                    try:
                        pred_result = predictor.predict(
                            title=custom_title.strip(),
                            genres=selected_genres,
                            custom_year=int(custom_year)
                        )
                    except Exception as err:
                        st.error(f"Error predicting rating: {err}")

    with col_result:
        st.subheader("Prediction")

        # Default prediction on first load
        if pred_result is None:
            try:
                pred_result = predictor.predict_by_title("Toy Story (1995)")
            except Exception:
                pred_result = None

        if pred_result:
            score = pred_result["predicted_rating"]
            verdict_text, verdict_class = get_verdict_info(score)

            # Rating Hero Card
            st.markdown(f"""
            <div class="rating-card">
                <div class="rating-label">Predicted Rating</div>
                <div class="rating-score">⭐ {score:.2f} <span style="font-size: 1.7rem; font-weight: 400; opacity: 0.85;">/ 5.0</span></div>
                <div class="verdict-tag {verdict_class}">{verdict_text}</div>
            </div>
            """, unsafe_allow_html=True)

            # Visual progress gauge
            gauge_val = min(1.0, max(0.0, (score - 0.5) / 4.5))
            st.progress(gauge_val, text=f"Rating Scale: {score:.2f} out of 5.0 stars")

            # Movie Overview Details Box
            st.markdown(f"""
            <div class="movie-info-box">
                <h4 style="margin-top: 0; color: #1a237e;">{pred_result['title']}</h4>
                <p style="margin-bottom: 8px;"><strong>Release Year:</strong> {pred_result['release_year']}</p>
                <div style="margin-bottom: 8px;">
                    {''.join([f"<span class='badge-genre'>{g}</span>" for g in pred_result['genres']]) if pred_result['genres'] else "<span class='badge-genre'>None</span>"}
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
