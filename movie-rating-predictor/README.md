# 🎬 Movie Rating Predictor

> **A beginner-friendly, mathematically sound Supervised Machine Learning Regression project using historical MovieLens data.**  
> *Designed specifically for college viva defense and learning foundational ML principles.*

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Project Structure](#-project-structure)
3. [Machine Learning Fundamentals (Explained Simply)](#-machine-learning-fundamentals-explained-simply)
   - [What is Machine Learning?](#1-what-is-machine-learning)
   - [What is Regression?](#2-what-is-regression)
   - [Features ($X$) and Target ($y$)](#3-features-x-and-target-y)
   - [Why Raw `userId` & `movieId` Are NOT Continuous Numbers](#4-why-raw-userid--movieid-are-not-continuous-numbers)
   - [Target Leakage: The Danger & Our Prevention Strategy](#5-target-leakage-the-danger--our-prevention-strategy)
   - [Training vs Testing Sets](#6-training-vs-testing-sets)
   - [What `fit()` and `predict()` Actually Do](#7-what-fit-and-predict-actually-do)
   - [How Linear Regression Works (OLS Intuition)](#8-how-linear-regression-works-ols-intuition)
   - [Why DummyRegressor Baseline Comparison is Mandatory](#9-why-dummyregressor-baseline-comparison-is-mandatory)
   - [Evaluation Metrics: MAE vs RMSE](#10-evaluation-metrics-mae-vs-rmse)
4. [Genuine Model Evaluation Results](#-genuine-model-evaluation-results)
5. [Model Interpretability & Feature Weights](#-model-interpretability--feature-weights)
6. [Quick Start & Installation Guide (Windows)](#-quick-start--installation-guide-windows)
7. [Streamlit Web Interface Walkthrough](#-streamlit-web-interface-walkthrough)
8. [College Viva Defense Guide (Questions & Answers)](#-college-viva-defense-guide-questions--answers)

---

## 🌟 Project Overview

The **Movie Rating Predictor** is an independent, end-to-end Machine Learning project that predicts the expected general rating of movies using the genuine **MovieLens dataset** (~9,742 movies and 100,836 ratings).

### Key Architectural Highlights:
- **Primary Model**: `LinearRegression` (Ordinary Least Squares) from `scikit-learn`.
- **Baseline Model**: `DummyRegressor(strategy='mean')` for scientific benchmarking.
- **Leakage Prevention**: Strictly splits data before computing summary statistics and uses **Leave-One-Out (Out-Of-Self)** target encoding for historical ratings on the training set.
- **Interpretability**: Complete transparency with explicit mathematical equations ($\hat{y} = \beta_0 + \sum \beta_i x_i$).
- **User Interface**: Interactive, modern Streamlit web application with real-time predictions, feature breakdowns, and evaluation dashboards.
- **100% Independent**: Kept completely isolated from any external applications.

---

## 📁 Project Structure

```text
movie-rating-predictor/
├── data/
│   ├── movies.csv               # 9,742 movies (movieId, title, genres)
│   └── ratings.csv              # 100,836 ratings (userId, movieId, rating, timestamp)
├── models/
│   ├── linear_regression_model.joblib      # Fitted scikit-learn LinearRegression model
│   ├── dummy_baseline_model.joblib         # Fitted scikit-learn DummyRegressor baseline
│   ├── preprocessing_artifacts.joblib      # Training statistics, median year, feature columns
│   ├── evaluation_metrics.json             # Genuine MAE and RMSE metrics on held-out test data
│   └── feature_coefficients.json           # Model intercept and feature weights
├── src/
│   ├── __init__.py
│   ├── train_model.py           # End-to-end ML training and evaluation pipeline
│   └── predict.py               # Standalone inference module & CLI prediction tool
├── app.py                       # Interactive Streamlit web interface
├── requirements.txt             # Python dependencies
└── README.md                    # Comprehensive documentation & college viva guide
```

---

## 🧠 Machine Learning Fundamentals (Explained Simply)

### 1. What is Machine Learning?
In traditional programming, a human writes explicit rules (if-else logic):
$$\text{Data} + \text{Handcrafted Rules} \longrightarrow \text{Output}$$

In **Machine Learning**, an algorithm learns mathematical relationships directly from historical data:
$$\text{Data} + \text{Known Outputs} \longrightarrow \text{Learned Model (Rules)}$$

Once trained, the learned model takes new, unseen data and predicts the outcome.

### 2. What is Regression?
Supervised Machine Learning is divided into two primary problem types:
- **Classification**: Predicting a discrete category or class label (e.g., *Spam vs. Not Spam*, *Cat vs. Dog*).
- **Regression**: Predicting a **continuous numerical quantity** on a scale (e.g., *Movie Rating from 0.5 to 5.0 stars*, *House Price in dollars*, *Temperature in degrees*).

Because movie ratings represent an ordered, continuous numerical scale, this project is a **regression** task.

### 3. Features ($X$) and Target ($y$)
- **Target ($y$)**: The variable we want to predict. Here, $y = \text{rating}$, a floating-point number between 0.5 and 5.0.
- **Features ($X$)**: The input measurements or attributes describing each sample that the model uses to make its prediction:
  1. `release_year`: Extracted from the movie title (e.g., `1995` from `"Toy Story (1995)"`).
  2. `genre_count`: Total number of genres tagged for the movie.
  3. `genre_<Name>`: 19 binary indicator columns (multi-hot encoding) for standard MovieLens genres (`Action`, `Animation`, `Comedy`, `Drama`, `Horror`, etc.).
  4. `hist_movie_avg_rating`: Historical rating reception computed strictly from training evidence with Bayesian smoothing.
  5. `log_rating_count`: Logarithm of historical rating frequency $\ln(1 + \text{count})$, measuring movie popularity.

### 4. Why Raw `userId` & `movieId` Are NOT Continuous Numbers
A frequent mistake made by beginners is feeding database IDs directly into a regression model:
```python
# ❌ INCORRECT (FATAL ML MISTAKE):
model.fit(df[['userId', 'movieId']], df['rating'])
```
#### Why is this invalid?
- `userId` and `movieId` are **nominal categorical identifiers** (arbitrary database keys).
- User #100 is **not** twice User #50. Movie #3000 is **not** three times better than Movie #1000.
- Feeding raw integer IDs into Linear Regression forces an absurd mathematical assumption: $\beta \times \text{userId}$, implying ratings monotonically increase or decrease just because an account registered later!
- **Our Solution**: We discard raw IDs and extract **meaningful intrinsic features** (genres, era, and statistical crowd reception).

### 5. Target Leakage: The Danger & Our Prevention Strategy
**Target Leakage** (or data leakage) occurs when information from the target variable ($y$) is inadvertently included in the feature set ($X$) during model training or preprocessing.

#### The Danger:
If you calculate each movie's average rating across the whole dataset and use that as an input feature:
- The rating $r_i$ being predicted is included in the average!
- The model merely learns the identity shortcut $\hat{y} \approx y$.
- Training error looks artificially near zero, but the model **fails catastrophically** when deployed on brand new data.

#### How We Prevented Target Leakage:
1. **Pre-Split Isolation**: We partitioned the dataset into 80% train and 20% test using `train_test_split(..., random_state=42)` **before** any aggregation was computed.
2. **Leave-One-Out (Out-Of-Self) Calculation**: In the training split, when sample $i$ with rating $r_i$ on movie $M$ is evaluated, its own rating is subtracted:
   $$\text{loo\_sum} = \text{train\_sum}_M - r_i$$
   $$\text{loo\_count} = \text{train\_count}_M - 1$$
   $$\text{hist\_avg} = \frac{\text{loo\_sum} + m \cdot \mu_{\text{global}}}{\text{loo\_count} + m}$$
   Row $i$'s rating is **never** allowed to predict itself!
3. **Held-Out Test Lookup**: For the test split, we strictly look up the precomputed training table. The test ratings are never seen by the feature extractor.

### 6. Training vs Testing Sets
- **Training Set (80% = 80,668 rows)**: Used by `model.fit()` to learn the mathematical coefficients.
- **Testing Set (20% = 20,168 rows)**: Completely held out in a locked vault. The model never sees these samples during training.
- **Reproducibility (`random_state=42`)**: Setting a fixed seed ensures that the pseudo-random split produces the identical train and test sets every time the script is executed.

### 7. What `fit()` and `predict()` Actually Do
- **`fit(X_train, y_train)`**:
  Solves the Ordinary Least Squares (OLS) closed-form matrix equation:
  $$\beta = (X^T X)^{-1} X^T y$$
  It searches for the exact vector of weights $\beta$ that minimizes the sum of squared differences between true targets $y_i$ and predictions $\hat{y}_i$.
- **`predict(X_test)`**:
  Applies the learned equation to new samples:
  $$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p$$
  No learning occurs during `predict()`; it is purely matrix multiplication.

### 8. How Linear Regression Works (OLS Intuition)
Linear Regression models the target as a weighted linear combination of inputs:
$$\hat{y} = \beta_0 + \sum_{j=1}^{p} \beta_j x_j$$
Where:
- $\beta_0$ (**Intercept**): The expected baseline rating when all numeric inputs are zero ($4.8264$).
- $\beta_j$ (**Coefficient/Slope**): The change in predicted rating for a 1-unit increase in feature $x_j$, holding all other features fixed.
- **Residual ($e_i$)**: The error on the $i$-th sample: $e_i = y_i - \hat{y}_i$. Ordinary Least Squares minimizes $\sum e_i^2$.

### 9. Why DummyRegressor Baseline Comparison is Mandatory
In genuine scientific machine learning, a model's score cannot be evaluated in a vacuum.
- Suppose a student presents: *"My model achieved an MAE of 0.83 stars."*
- A professor will ask: *"Is that good? What if I didn't use machine learning at all?"*
- A **`DummyRegressor(strategy='mean')`** ignores every feature and simply outputs the dataset average ($\approx 3.50$ stars) for every movie.
- On our test set, this trivial guess achieves an MAE of **0.8316**.
- By comparing our `LinearRegression` model (MAE = **0.7483**) against `DummyRegressor`, we mathematically prove that our features provide a genuine **+10.01% error reduction** over naive guessing.

### 10. Evaluation Metrics: MAE vs RMSE
| Metric | Mathematical Formula | Meaning in College Viva |
| :--- | :--- | :--- |
| **MAE** (Mean Absolute Error) | $\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} \|y_i - \hat{y}_i\|$ | *"On average, our prediction is off by 0.748 stars."* Treats all errors linearly. |
| **RMSE** (Root Mean Squared Error) | $\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$ | *"Penalizes large errors much more severely."* Because errors are squared before averaging, a 2-star mistake counts 4 times as heavily as a 1-star mistake. |

---

## 📊 Genuine Model Evaluation Results

These metrics were genuinely calculated on the 20,168 held-out test ratings:

```text
----------------------------------------------------------------------
[EVALUATION] MODEL EVALUATION RESULTS (HELD-OUT TEST SET)
----------------------------------------------------------------------
Total Ratings in Dataset : 100,836
Training Set Size        : 80,668 rows (80%)
Held-Out Test Set Size   : 20,168 rows (20%)
----------------------------------------------------------------------
Metric                    | Dummy Baseline     | Linear Regression  | Improvement 
----------------------------------------------------------------------
MAE (Mean Absolute Error) | 0.8316 stars       | 0.7483 stars       | +10.01%
RMSE (Root Mean Sq Error) | 1.0488 stars       | 0.9582 stars       | +8.65%
----------------------------------------------------------------------
```

---

## 🔍 Model Interpretability & Feature Weights

Because we used Linear Regression, we can directly inspect how each feature influences a movie's predicted rating:

**Model Intercept ($\beta_0$)**: `+4.8264`

| Feature | Coefficient ($\beta$) | Real-World Interpretation |
| :--- | :--- | :--- |
| `hist_movie_avg_rating` | `+0.9140` | Strongest predictor: prior crowd reception reliably carries over to future viewers. |
| `log_rating_count` | `+0.0746` | Frequently rated, popular movies tend to receive slightly higher ratings. |
| `genre_Documentary` | `+0.3610` | Consistently rated higher due to high viewer self-selection. |
| `genre_Animation` | `+0.1285` | High average appreciation on MovieLens. |
| `genre_Drama` | `+0.0561` | Critical acclaim associated with drama movies. |
| `genre_Film-Noir` | `+0.0658` | Niche classic appeal. |
| `genre_Children` | `-0.1509` | Tends to receive harsher ratings from general adult MovieLens reviewers. |
| `genre_Horror` | `-0.1123` | Polarizing genre with higher proportion of critical reviews. |
| `genre_Action` | `-0.0588` | High-volume commercial genre with wide variance. |
| `genre_Comedy` | `-0.0574` | Subjective humor leads to lower average scores than serious drama. |
| `release_year` | `-0.0024` | Slight classic survivorship bias (older movies in catalog are celebrated classics). |

---

## 🚀 Quick Start & Installation Guide (Windows)

### Step 1: Open PowerShell or Terminal
Navigate to the project directory:
```powershell
cd C:\Users\Dilip\Desktop\Projects\movie-rating-predictor
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Training Pipeline
Execute the genuine training script:
```powershell
python src\train_model.py
```
*This will ingest `movies.csv` and `ratings.csv`, build features without leakage, train `LinearRegression` and `DummyRegressor`, evaluate on the 20% test split, and save models into `models/`.*

### Step 4: Run a Prediction from CLI
You can test predictions directly from the command line:
```powershell
# Predict for an existing movie in MovieLens:
python src\predict.py "Toy Story (1995)"
python src\predict.py "Godfather, The (1972)"

# Predict for a brand new, custom movie:
python src\predict.py "Space Frontier" --genres "Action|Sci-Fi" --year 2025
```

### Step 5: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ CinePredict Web Application

The interactive web application is designed as a production-ready platform with three main sections:

1. **🔮 Predict Movie Rating (Studio)**:
   - **Catalog Mode**: Search and select from 9,700+ movies with instant auto-fill and metadata previews.
   - **Custom Pitch Mode**: Enter any new, indie, or upcoming movie title, release year, and genre mix.
   - **Live Prediction Card**: Large Star Rating badge (e.g., `⭐ 3.95 / 5.0`) with visual meter and audience reception verdict (*Critical Masterpiece*, *Highly Recommended*, *Solid Watch*, etc.).
   - **Key Influencing Factors**: Shows which genres and historical patterns raised or lowered the expected score.

2. **📈 Genre & Market Insights**:
   - Genre Appreciation Index showing which genres statistically perform best with viewers.
   - Real-world distribution across 100,000+ ratings and 9,742 films.

3. **ℹ️ About the Model**:
   - Clean, professional breakdown of the engine, methodology, and performance benchmarks.

---

## 🎓 College Viva Defense Guide (Questions & Answers)

### Q1: What type of Machine Learning problem is this?
> **Answer**: *"This is a **Supervised Machine Learning Regression** task. It is supervised because our training data contains both input features $X$ and known ground-truth targets $y$. It is regression because our target variable (movie rating) is a continuous numerical score between 0.5 and 5.0 stars, rather than a discrete classification category."*

### Q2: Why did you choose Linear Regression instead of Random Forest?
> **Answer**: *"We chose Linear Regression for three primary reasons:
> 1. **Interpretability**: In Linear Regression, the formula $\hat{y} = \beta_0 + \sum \beta_i x_i$ is completely transparent. We can explain exactly how much each genre and feature shifts the prediction.
> 2. **Foundational Best Practice**: Professional ML methodology dictates establishing a linear baseline before jumping into complex non-linear models.
> 3. **Viva Clarity**: Unlike black-box ensemble methods like Random Forest, Linear Regression allows us to mathematically explain the optimization process (Ordinary Least Squares) from first principles."*

### Q3: Why didn't you feed `userId` and `movieId` as numerical features?
> **Answer**: *"Because `userId` and `movieId` are **nominal categorical variables** (arbitrary database identifiers). Treating them as continuous numbers implies that User #200 is double User #100, which has no mathematical basis. Instead, we extracted intrinsic movie features: multi-hot genre vectors, release era, and statistical crowd reception from the training split."*

### Q4: What is Target Leakage and how did you prevent it?
> **Answer**: *"Target leakage occurs when information from the target variable is inadvertently present in input features during training. If we computed a movie's average rating using the whole dataset, a rating being predicted would be part of its own input feature, causing artificial overfitting.
> We prevented this by:
> 1. Splitting data into 80% train and 20% test **before** computing historical aggregations.
> 2. Using **Leave-One-Out (Out-Of-Self)** target encoding on the training set: for every sample $i$, we subtract its rating $r_i$ from the movie's running sum so it never predicts itself.
> 3. Evaluating the test set strictly using precomputed training lookups."*

### Q5: What do `fit()` and `predict()` do mathematically?
> **Answer**: *"`fit(X, y)` solves Ordinary Least Squares by minimizing the residual sum of squares: $\beta = (X^T X)^{-1} X^T y$. `predict(X)` simply applies the learned weights to new inputs via matrix multiplication: $\hat{y} = X \beta$."*

### Q6: Why did you include DummyRegressor?
> **Answer**: *"Evaluating an ML model in isolation is unscientific. If our model achieves an MAE of 0.7483 stars, we must prove that this error is significantly better than a trivial baseline. DummyRegressor always predicts the training average (3.50 stars) and achieves an MAE of 0.8316 stars. Beating it by 10.01% proves our model has learned genuine predictive patterns."*

### Q7: What is the difference between MAE and RMSE?
> **Answer**: *"MAE represents the average absolute error ($0.7483$ stars). RMSE squares errors before taking the root ($0.9582$ stars), which penalizes large blunders disproportionately. When RMSE is substantially higher than MAE, it indicates the model made a few large errors rather than many small ones."*

---

## 📜 Academic Integrity & License
This project was developed for educational and viva defense purposes using the publicly available MovieLens dataset from GroupLens Research.
