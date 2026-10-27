"""
Movie Rating Predictor - Web Application
========================================
A modern, dark cinematic web application powered by machine learning 
to predict audience movie ratings.
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

# Dark cinematic & glassmorphism styling
st.markdown("""
<style>
    /* Hide Streamlit Header, Deploy Button, MainMenu, and Footer */
    #MainMenu {
        visibility: hidden !important;
        display: none !important;
    }
    header[data-testid="stHeader"], header {
        visibility: hidden !important;
        display: none !important;
        height: 0px !important;
    }
    footer {
        visibility: hidden !important;
        display: none !important;
    }
    .stDeployButton {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }
    [data-testid="stDecoration"] {
        display: none !important;
    }
    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* Global Background & Font */
    .stApp {
        background: radial-gradient(circle at 50% 20%, #151928 0%, #0d101a 60%, #07090f 100%);
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Top Header */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 2px;
        text-shadow: 0 0 25px rgba(129, 140, 248, 0.35);
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Section Subheadings */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 1rem;
    }

    /* Glassmorphism Card (Left: Movie Information) */
    .glass-card {
        background: rgba(19, 24, 38, 0.72);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 24px 26px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }

    /* Neon Gradient Prediction Card (Right) */
    .prediction-outer-glow {
        padding: 2px;
        border-radius: 20px;
        background: linear-gradient(135deg, #06b6d4 0%, #6366f1 50%, #f43f5e 100%);
        box-shadow: 0 0 24px rgba(6, 182, 212, 0.35), 0 0 35px rgba(244, 63, 94, 0.25);
        margin-bottom: 18px;
    }

    .prediction-card-inner {
        background: #0f1423;
        border-radius: 18px;
        padding: 26px 20px;
        text-align: center;
    }

    .prediction-label {
        font-size: 1.05rem;
        color: #cbd5e1;
        font-weight: 500;
        margin-bottom: 14px;
    }

    .rating-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        margin-bottom: 16px;
    }

    .star-icon {
        font-size: 2.8rem;
        filter: drop-shadow(0 0 16px rgba(251, 191, 36, 0.65));
    }

    .rating-score {
        font-size: 3.4rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
        letter-spacing: -1px;
    }

    .rating-max {
        font-size: 1.6rem;
        font-weight: 400;
        color: #64748b;
        margin-left: 4px;
    }

    .verdict-pill {
        display: inline-block;
        padding: 7px 22px;
        border-radius: 30px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.2px;
    }

    /* Rating Scale / Progress */
    .scale-wrapper {
        margin-top: 4px;
        margin-bottom: 18px;
    }

    .scale-text {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-bottom: 6px;
    }

    .scale-bar-bg {
        width: 100%;
        height: 10px;
        background: #1e2638;
        border-radius: 6px;
        overflow: hidden;
    }

    .scale-bar-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #06b6d4 0%, #8b5cf6 55%, #ec4899 100%);
        box-shadow: 0 0 12px rgba(139, 92, 246, 0.5);
    }

    /* Movie Details Card (Bottom Right) */
    .details-card {
        background: rgba(19, 24, 38, 0.72);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 22px 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }

    .details-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }

    .details-year {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 14px;
    }

    /* Vibrant Genre Badges */
    .genre-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #ffffff;
        margin-right: 6px;
        margin-bottom: 6px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* Input Glow & Radios */
    div[data-baseweb="select"] > div {
        background-color: rgba(15, 20, 35, 0.85) !important;
        border: 1.5px solid #38bdf8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.3) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
    }

    div[data-baseweb="input"] > div {
        background-color: rgba(15, 20, 35, 0.85) !important;
        border: 1.5px solid rgba(56, 189, 248, 0.6) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
    }

    div[role="radiogroup"] {
        display: flex;
        gap: 12px;
        margin-bottom: 10px;
    }

    div[role="radiogroup"] label {
        color: #cbd5e1 !important;
    }

    /* Coral-Red Glowing Predict Rating Button */
    div.stButton > button {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff6464 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 20px rgba(255, 75, 75, 0.45) !important;
        transition: all 0.25s ease !important;
        margin-top: 14px !important;
    }

    div.stButton > button:hover {
        box-shadow: 0 6px 26px rgba(255, 75, 75, 0.7) !important;
        transform: translateY(-1px) !important;
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


def get_genre_pill_html(genre: str) -> str:
    """
    Return a vibrant gradient pill HTML for a given movie genre.
    """
    palette = {
        "Adventure": "linear-gradient(135deg, #0284c7, #2563eb)",
        "Children": "linear-gradient(135deg, #6366f1, #4f46e5)",
        "Fantasy": "linear-gradient(135deg, #c026d3, #9333ea)",
        "Action": "linear-gradient(135deg, #ef4444, #dc2626)",
        "Animation": "linear-gradient(135deg, #f59e0b, #d97706)",
        "Comedy": "linear-gradient(135deg, #10b981, #059669)",
        "Crime": "linear-gradient(135deg, #64748b, #475569)",
        "Documentary": "linear-gradient(135deg, #14b8a6, #0d9488)",
        "Drama": "linear-gradient(135deg, #f97316, #ea580c)",
        "Horror": "linear-gradient(135deg, #be123c, #9f1239)",
        "Mystery": "linear-gradient(135deg, #7c3aed, #6d28d9)",
        "Romance": "linear-gradient(135deg, #ec4899, #db2777)",
        "Sci-Fi": "linear-gradient(135deg, #06b6d4, #0891b2)",
        "Thriller": "linear-gradient(135deg, #e11d48, #be123c)",
        "IMAX": "linear-gradient(135deg, #3b82f6, #1d4ed8)",
        "Western": "linear-gradient(135deg, #b45309, #78350f)",
        "War": "linear-gradient(135deg, #475569, #334155)",
        "Musical": "linear-gradient(135deg, #a855f7, #7e22ce)",
        "Film-Noir": "linear-gradient(135deg, #334155, #1e293b)",
    }
    bg = palette.get(genre, "linear-gradient(135deg, #6366f1, #4f46e5)")
    return f"<span class='genre-pill' style='background: {bg};'>{genre}</span>"


def get_verdict_info(rating: float) -> Tuple[str, str, str]:
    """
    Return human-readable verdict, background rgba, and border rgba for the rating pill.
    """
    if rating >= 4.2:
        return (
            "Must Watch / Critical Masterpiece 🏆",
            "rgba(234, 179, 8, 0.12)",
            "rgba(250, 204, 21, 0.8)",
            "#facc15"
        )
    elif rating >= 3.7:
        return (
            "Highly Recommended / Strong Reception ⭐",
            "rgba(34, 197, 94, 0.12)",
            "rgba(74, 222, 128, 0.8)",
            "#4ade80"
        )
    elif rating >= 3.2:
        return (
            "Decent Watch / Positive Audience Score 🍿",
            "rgba(14, 165, 233, 0.12)",
            "rgba(56, 189, 248, 0.8)",
            "#38bdf8"
        )
    elif rating >= 2.8:
        return (
            "Mixed Reactions / Polarizing ⚖️",
            "rgba(249, 115, 22, 0.12)",
            "rgba(251, 146, 60, 0.8)",
            "#fb923c"
        )
    else:
        return (
            "Below Average / Niche ⚠️",
            "rgba(239, 68, 68, 0.12)",
            "rgba(248, 113, 113, 0.8)",
            "#f87171"
        )


def main():
    # Top Header
    st.markdown("""
    <div style="margin-bottom: 1.8rem;">
        <div class="hero-title">🎬 MOVIE RATING PREDICTOR</div>
        <div class="hero-subtitle">Predict audience ratings based on metadata and genre profile.</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(
            "⚠️ **Model Artifacts Not Found!** "
            f"Please run `python src/train_model.py` first. Error: {e}"
        )
        return

    # Initialize session state for persistent prediction
    if "predicted_data" not in st.session_state:
        try:
            st.session_state["predicted_data"] = predictor.predict_by_title("Jumanji (1995)")
            st.session_state["predicted_signature"] = ("catalog", "Jumanji (1995)")
        except Exception:
            st.session_state["predicted_data"] = None
            st.session_state["predicted_signature"] = None

    # Two column layout: Left for inputs, Right for predictions
    col_input, col_result = st.columns([1.1, 1], gap="large")

    current_signature = None
    current_movie_info = None

    with col_input:
        st.markdown('<div class="section-title">Movie Information</div>', unsafe_allow_html=True)

        input_mode = st.radio(
            "Select Input Type:",
            ["Choose from Catalog (9,700+ Movies)", "Enter a Custom / New Movie"],
            horizontal=True
        )

        if input_mode == "Choose from Catalog (9,700+ Movies)":
            if predictor.movies_df is not None and not predictor.movies_df.empty:
                movie_titles = predictor.movies_df["title"].tolist()
                
                # Check session state for selected title
                if "catalog_selected_title" not in st.session_state or st.session_state["catalog_selected_title"] not in movie_titles:
                    default_idx = 0
                    if "Jumanji (1995)" in movie_titles:
                        default_idx = movie_titles.index("Jumanji (1995)")
                else:
                    default_idx = movie_titles.index(st.session_state["catalog_selected_title"])

                selected_title = st.selectbox(
                    "Search or Select Movie Title:",
                    options=movie_titles,
                    index=default_idx,
                    key="catalog_selected_title"
                )

                # Fetch metadata for selected movie
                matched = predictor.movies_df[predictor.movies_df["title"] == selected_title]
                if not matched.empty:
                    current_genres = [g for g in str(matched.iloc[0]["genres"]).split("|") if g and g != "(no genres listed)"]
                    current_year = predictor._extract_year(selected_title)
                else:
                    current_genres = []
                    current_year = int(predictor.median_year)

                current_signature = ("catalog", selected_title)
                current_movie_info = {
                    "title": selected_title,
                    "release_year": int(current_year),
                    "genres": current_genres
                }

                # Preview badges on left
                if current_genres:
                    badges_preview = "".join([get_genre_pill_html(g) for g in current_genres])
                    st.markdown(f"<div style='margin-top: 8px; margin-bottom: 10px;'>{badges_preview}</div>", unsafe_allow_html=True)

                if st.button("Predict Rating", type="primary", use_container_width=True):
                    try:
                        res = predictor.predict_by_title(selected_title)
                        st.session_state["predicted_data"] = res
                        st.session_state["predicted_signature"] = current_signature
                        st.toast(f"✅ Predicted {res['predicted_rating']:.2f}★ for {selected_title}!", icon="🎬")
                    except Exception as err:
                        st.error(f"Error predicting rating: {err}")

            else:
                st.warning("Movie catalog is unavailable.")

        else:
            # Custom movie inputs
            custom_title = st.text_input("Movie Title", value="Project Odyssey", placeholder="e.g. Inception 2", key="custom_title_input")
            custom_year = st.number_input("Release Year", min_value=1920, max_value=2030, value=2025, step=1, key="custom_year_input")

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

            # Show live pills of selected genres
            if selected_genres:
                custom_badges = "".join([get_genre_pill_html(g) for g in selected_genres])
                st.markdown(f"<div style='margin-top: 6px; margin-bottom: 12px;'>{custom_badges}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='margin-top: 6px; margin-bottom: 12px; color: #64748b; font-size: 0.85rem;'>No genres selected (neutral baseline applied)</div>", unsafe_allow_html=True)

            display_title = custom_title.strip() if custom_title.strip() else "Untitled Custom Movie"
            current_signature = ("custom", display_title, int(custom_year), tuple(sorted(selected_genres)))
            current_movie_info = {
                "title": display_title,
                "release_year": int(custom_year),
                "genres": selected_genres
            }

            if st.button("Predict Rating", type="primary", use_container_width=True):
                try:
                    res = predictor.predict(
                        title=display_title,
                        genres=selected_genres,
                        custom_year=int(custom_year)
                    )
                    st.session_state["predicted_data"] = res
                    st.session_state["predicted_signature"] = current_signature
                    st.toast(f"✅ Predicted {res['predicted_rating']:.2f}★ for {display_title}!", icon="🎬")
                except Exception as err:
                    st.error(f"Error predicting rating: {err}")

    with col_result:
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

        is_up_to_date = (st.session_state.get("predicted_signature") == current_signature)
        pred_data = st.session_state.get("predicted_data")

        if is_up_to_date and pred_data is not None:
            # Active, computed prediction
            score = pred_data["predicted_rating"]
            verdict_text, verdict_bg, verdict_border, verdict_color = get_verdict_info(score)

            # Neon Glow Hero Card
            st.markdown(f"""
            <div class="prediction-outer-glow">
                <div class="prediction-card-inner">
                    <div class="prediction-label">Predicted Rating</div>
                    <div class="rating-row">
                        <span class="star-icon">⭐</span>
                        <span class="rating-score">{score:.2f}</span>
                        <span class="rating-max">/ 5.0</span>
                    </div>
                    <div>
                        <span class="verdict-pill" style="background: {verdict_bg}; border: 1px solid {verdict_border}; color: {verdict_color}; box-shadow: 0 0 14px {verdict_bg};">
                            {verdict_text}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Rating Scale with Gradient Bar
            fill_pct = min(100.0, max(0.0, (score / 5.0) * 100.0))
            st.markdown(f"""
            <div class="scale-wrapper">
                <div class="scale-text">Rating Scale: {score:.2f} out of 5.0 stars</div>
                <div class="scale-bar-bg">
                    <div class="scale-bar-fill" style="width: {fill_pct:.1f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Movie Overview Details Box
            genres_html = "".join([get_genre_pill_html(g) for g in pred_data['genres']]) if pred_data['genres'] else "<span class='genre-pill' style='background: #475569;'>None</span>"
            st.markdown(f"""
            <div class="details-card">
                <div class="details-title">{pred_data['title']}</div>
                <div class="details-year">Release Year: {pred_data['release_year']}</div>
                <div>{genres_html}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Standby state - waiting for user to click "Predict Rating"
            target_info = current_movie_info if current_movie_info else {"title": "Selected Movie", "release_year": "—", "genres": []}
            
            st.markdown(f"""
            <div class="prediction-outer-glow" style="background: linear-gradient(135deg, rgba(6, 182, 212, 0.3) 0%, rgba(99, 102, 241, 0.3) 50%, rgba(244, 63, 94, 0.3) 100%); box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);">
                <div class="prediction-card-inner">
                    <div class="prediction-label">Awaiting Prediction</div>
                    <div class="rating-row">
                        <span class="star-icon" style="filter: drop-shadow(0 0 8px rgba(148, 163, 184, 0.3)); opacity: 0.65;">⭐</span>
                        <span class="rating-score" style="color: #64748b; letter-spacing: 2px;">?.??</span>
                        <span class="rating-max">/ 5.0</span>
                    </div>
                    <div>
                        <span class="verdict-pill" style="background: rgba(255, 75, 75, 0.12); border: 1.5px dashed rgba(255, 75, 75, 0.7); color: #ff8585; box-shadow: 0 0 12px rgba(255, 75, 75, 0.25);">
                            👉 Click "Predict Rating" to Run Model
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Rating Scale (Empty/Standby)
            st.markdown("""
            <div class="scale-wrapper">
                <div class="scale-text" style="color: #64748b;">Rating Scale: Ready to calculate...</div>
                <div class="scale-bar-bg">
                    <div class="scale-bar-fill" style="width: 0%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Movie Overview Details Box showing configured movie
            genres_html = "".join([get_genre_pill_html(g) for g in target_info['genres']]) if target_info['genres'] else "<span class='genre-pill' style='background: #475569;'>None selected</span>"
            st.markdown(f"""
            <div class="details-card">
                <div class="details-title">{target_info['title']}</div>
                <div class="details-year">Release Year: {target_info['release_year']}</div>
                <div>{genres_html}</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
