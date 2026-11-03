
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px
from datetime import date, time

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="YouTube AI Popularity Predictor",
    page_icon="▶️",
    layout="wide"
)

@st.cache_resource
def load_models():
    pre = joblib.load(BASE + "/models/pre_upload_model.pkl")
    early = joblib.load(BASE + "/models/early_performance_model.pkl")
    with open(BASE + "/models/metadata.json") as f:
        meta = json.load(f)
    return pre, early, meta

@st.cache_data
def load_data():
    return pd.read_csv(BASE + "/dataset/cleaned_youtube_data.csv")

pre_model, early_model, meta = load_models()
data = load_data()

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
[data-testid="stMetric"] {
    background-color: #1c2333;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("▶️ YouTube Video Popularity Prediction")
st.caption("Artificial Intelligence and Machine Learning Project")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Home",
        "Dataset Analytics",
        "Pre-Upload Prediction",
        "Early Performance Prediction",
        "Model Performance",
        "About Project"
    ]
)

# ---------------- HOME ----------------

if page == "Home":

    st.header("Welcome to YouTube AI Analytics")

    st.write(
        "An AI-powered application that analyzes YouTube "
        "video characteristics and predicts popularity."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Dataset Records", f"{len(data):,}")
    c2.metric("Unique Videos", f"{data.video_id.nunique():,}")
    c3.metric("Categories", data.category_id.nunique())
    c4.metric("Prediction Modes", "2")

    st.subheader("Project Features")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**Pre-Upload Prediction**\n\n"
            "Uses video metadata such as title, category, tags "
            "and description."
        )

    with col2:
        st.success(
            "**Early Performance Prediction**\n\n"
            "Uses initial trending views, likes and comments "
            "to predict later popularity."
        )

    st.subheader("Popularity Distribution")

    popularity_counts = data.groupby("video_id")["views"].max().reset_index()

    fig = px.histogram(
        popularity_counts,
        x="views",
        nbins=40,
        title="Distribution of Maximum Recorded Video Views"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- DATA ANALYTICS ----------------

elif page == "Dataset Analytics":

    st.header("Dataset Analytics")

    st.write("Explore the YouTube India trending dataset.")

    st.subheader("Dataset Preview")
    st.dataframe(data.head(100), use_container_width=True)

    st.subheader("Category Distribution")

    cat = (
        data.groupby("category_name")
        .size()
        .reset_index(name="Videos")
        .sort_values("Videos", ascending=False)
    )

    fig = px.bar(
        cat,
        x="category_name",
        y="Videos",
        color="Videos",
        title="Videos by Category"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Views vs Likes")

    fig = px.scatter(
        data.sample(min(3000, len(data)), random_state=42),
        x="views",
        y="likes",
        color="category_name",
        hover_data=["title"],
        title="Views and Likes Relationship"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Videos by Views")

    top = (
        data.sort_values("views", ascending=False)
        [["title", "channel_title", "views", "likes", "comment_count"]]
        .head(10)
    )

    st.dataframe(top, use_container_width=True)

# ---------------- PRE-UPLOAD PREDICTION ----------------

elif page == "Pre-Upload Prediction":

    st.header("Predict Popularity Before Upload")

    st.write(
        "Enter video information. Views, likes and comments "
        "are deliberately excluded because the video has not "
        "been published yet."
    )

    with st.form("pre_form"):

        title = st.text_input(
            "Video Title",
            "My New YouTube Video"
        )

        description = st.text_area(
            "Video Description",
            "This is my new video."
        )

        tags = st.text_area(
            "Video Tags (separate using |)",
            "youtube|trending|video"
        )

        categories = meta["categories"]

        category_names = list(categories.values())

        category = st.selectbox(
            "Video Category",
            category_names
        )

        publish_date = st.date_input(
            "Planned Publishing Date",
            date.today()
        )

        publish_time = st.time_input(
            "Planned Publishing Time",
            time(18, 0)
        )

        submitted = st.form_submit_button(
            "Predict Video Popularity",
            use_container_width=True
        )

    if submitted:

        if not title.strip():
            st.error("Please enter a video title.")
        else:

            category_id = next(
                (
                    k for k, v in categories.items()
                    if v == category
                ),
                "0"
            )

            tag_count = len([
                t for t in tags.split("|") if t.strip()
            ])

            features = pd.DataFrame([{
                "title": title,
                "tags": tags,
                "description": description,
                "category_id": str(category_id),
                "title_length": len(title),
                "tag_count": tag_count,
                "description_length": len(description),
                "publish_weekday": publish_date.weekday(),
                "publish_hour": publish_time.hour
            }])

            prediction = pre_model.predict(features)[0]

            probabilities = pre_model.predict_proba(features)[0]

            st.subheader("Prediction Result")

            colors = {
                "Low": "🔴",
                "Medium": "🟡",
                "High": "🟢"
            }

            st.markdown(
                f"## {colors[prediction]} {prediction} Popularity"
            )

            st.write("Model confidence for the predicted class:")

            st.progress(float(max(probabilities)))

            st.write(
                f"{max(probabilities) * 100:.2f}%"
            )

            st.caption(
                "This is a model probability, not a guarantee "
                "of actual YouTube success."
            )

            st.subheader("Class Probabilities")

            probability_df = pd.DataFrame({
                "Popularity": pre_model.classes_,
                "Probability": probabilities
            })

            fig = px.bar(
                probability_df,
                x="Popularity",
                y="Probability",
                color="Popularity",
                range_y=[0, 1]
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Content Suggestions")

            suggestions = []

            if len(title) < 25:
                suggestions.append(
                    "Consider making the title more descriptive."
                )

            if tag_count < 3:
                suggestions.append(
                    "Consider adding a few relevant tags."
                )

            if len(description) < 100:
                suggestions.append(
                    "Consider writing a more informative description."
                )

            if suggestions:
                for suggestion in suggestions:
                    st.write("💡", suggestion)
            else:
                st.write(
                    "Your metadata contains the basic recommended "
                    "elements. Continue improving content quality."
                )

# ---------------- EARLY PERFORMANCE ----------------

elif page == "Early Performance Prediction":

    st.header("Early Performance Prediction")

    st.warning(
        "This model was trained on the first available trending "
        "observation of historical videos. It is not a first-hour "
        "prediction model."
    )

    with st.form("early_form"):

        title = st.text_input(
            "Video Title",
            "My New YouTube Video"
        )

        description = st.text_area(
            "Description",
            "My video description"
        )

        tags = st.text_input(
            "Tags separated by |",
            "youtube|video|trending"
        )

        categories = meta["categories"]

        category = st.selectbox(
            "Category",
            list(categories.values()),
            key="early_category"
        )

        publish_date = st.date_input(
            "Video Publishing Date",
            date.today(),
            key="early_date"
        )

        publish_time = st.time_input(
            "Video Publishing Time",
            time(12, 0),
            key="early_time"
        )

        st.subheader("Initial Trending Statistics")

        views = st.number_input(
            "Initial Views",
            min_value=0,
            value=1000
        )

        likes = st.number_input(
            "Initial Likes",
            min_value=0,
            value=100
        )

        dislikes = st.number_input(
            "Initial Dislikes",
            min_value=0,
            value=5
        )

        comments = st.number_input(
            "Initial Comments",
            min_value=0,
            value=10
        )

        observation_date = st.date_input(
            "Date of Initial Observation",
            date.today(),
            key="observation_date"
        )

        submitted = st.form_submit_button(
            "Predict Future Popularity",
            use_container_width=True
        )

    if submitted:

        category_id = next(
            (
                k for k, v in categories.items()
                if v == category
            ),
            "0"
        )

        tag_count = len([
            t for t in tags.split("|") if t.strip()
        ])

        age_days = max(
            0,
            (observation_date - publish_date).days
        )

        features = pd.DataFrame([{
            "title": title,
            "tags": tags,
            "description": description,
            "category_id": str(category_id),
            "title_length": len(title),
            "tag_count": tag_count,
            "description_length": len(description),
            "publish_weekday": publish_date.weekday(),
            "publish_hour": publish_time.hour,
            "log_views": np.log1p(views),
            "log_likes": np.log1p(likes),
            "log_dislikes": np.log1p(dislikes),
            "log_comments": np.log1p(comments),
            "age_days": age_days,
            "engagement_rate": (likes + comments) / (views + 1)
        }])

        prediction = early_model.predict(features)[0]

        probabilities = early_model.predict_proba(features)[0]

        st.subheader("Future Popularity Prediction")

        st.markdown(f"## Predicted: {prediction}")

        st.metric(
            "Model Confidence",
            f"{max(probabilities) * 100:.2f}%"
        )

        prob_df = pd.DataFrame({
            "Popularity": early_model.classes_,
            "Probability": probabilities
        })

        fig = px.bar(
            prob_df,
            x="Popularity",
            y="Probability",
            color="Popularity",
            range_y=[0, 1]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "The prediction estimates the popularity class "
            "based on the historical dataset's final recorded "
            "view distribution."
        )

# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    st.header("Machine Learning Model Performance")

    for name in ["Pre Upload", "Early Performance"]:

        st.subheader(name)

        path = BASE + "/results/" + name + "_comparison.csv"

        results = pd.read_csv(path)

        st.dataframe(
            results.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}"
            }),
            use_container_width=True
        )

        st.image(
            BASE + "/results/" + name + "_comparison.png",
            use_container_width=True
        )

        st.image(
            BASE + "/results/" + name + "_confusion.png",
            use_container_width=True
        )

# ---------------- ABOUT ----------------

elif page == "About Project":

    st.header("About This Project")

    st.markdown("""
    ### Project Title
    YouTube Video Popularity Prediction Using Artificial
    Intelligence and Machine Learning.

    ### Technologies
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - TF-IDF
    - Logistic Regression
    - Random Forest
    - Streamlit
    - Plotly

    ### Project Objectives
    1. Analyze historical YouTube trending data.
    2. Identify factors associated with video popularity.
    3. Train machine learning classification models.
    4. Predict video popularity before upload.
    5. Predict eventual popularity using early trending data.

    ### Dataset
    YouTube India Trending Videos dataset.

    ### Popularity Classes
    - Low
    - Medium
    - High

    Classes are based on the dataset's one-third and
    two-thirds maximum-view quantiles.

    ### Limitations
    - Historical Indian trending data may not represent
      all YouTube users or future trends.
    - Pre-upload prediction is an estimate based on
      metadata associations.
    - Early prediction uses trending observations,
      not arbitrary first-hour statistics.
    - Predictions do not guarantee viral success.
    """)

    st.success("Thank you for exploring the project!")
