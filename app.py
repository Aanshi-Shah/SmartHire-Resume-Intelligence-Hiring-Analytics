import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="HR Resume Analytics Dashboard",
                   layout="wide")


st.title("📊 SmartHire : Resume Intelligence & Hiring Analytics")
st.markdown("Professional Resume Data Analysis & Prediction System")


df = pd.read_csv("resume.csv")

st.sidebar.header("Filter Candidates")

min_exp = int(df["Experience (Years)"].min())
max_exp = int(df["Experience (Years)"].max())

exp_filter = st.sidebar.slider(
    "Select Experience Range",
    min_exp, max_exp, (min_exp, max_exp)
)

df = df[(df["Experience (Years)"] >= exp_filter[0]) &
        (df["Experience (Years)"] <= exp_filter[1])]


# KPI CARDS

total_candidates = len(df)
avg_score = round(df["Resume Score(0-100)"].mean(), 2)
avg_experience = round(df["Experience (Years)"].mean(), 2)

col1, col2, col3 = st.columns(3)

col1.metric("Total Candidates", total_candidates)
col2.metric("Average Resume Score", avg_score)
col3.metric("Average Experience (Years)", avg_experience)

st.markdown("---")


# ROW 1 - DISTRIBUTION + PIE

col4, col5 = st.columns(2)

with col4:
    st.subheader("Resume Score Distribution")
    fig1 = px.histogram(df,
                        x="Resume Score(0-100)",
                        nbins=20,
                        color_discrete_sequence=["#4CAF50"])
    st.plotly_chart(fig1, use_container_width=True)

with col5:
    st.subheader("Experience Distribution")
    exp_counts = df["Experience (Years)"].value_counts()
    fig2 = px.pie(values=exp_counts.values,
                  names=exp_counts.index,
                  hole=0.4)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")


# CORRELATION ANALYSIS

st.subheader("Feature Correlation with Resume Score")

corr = df.corr(numeric_only=True)["Resume Score(0-100)"]
corr = corr.drop("Resume Score(0-100)")
corr = corr.sort_values()

fig3 = px.bar(x=corr.values,
              y=corr.index,
              orientation='h',
              color=corr.values,
              color_continuous_scale="Blues")

st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")


# MACHINE LEARNING MODEL

X = df.drop("Resume Score(0-100)", axis=1)
y = df["Resume Score(0-100)"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Feature Importance
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

st.subheader("Top Features Affecting Resume Score")

fig4 = px.bar(importance_df.tail(10),
              x="Importance",
              y="Feature",
              orientation='h',
              color="Importance",
              color_continuous_scale="Viridis")

st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# PREDICTOR SECTION (FIXED)

st.subheader("🎯 Resume Score Predictor")

exp_input = st.slider("Experience Years", min_exp, max_exp, 2)
tech_input = st.slider("Technical Score", 0, 100, 50)
soft_input = st.slider("Soft Skills Score", 0, 100, 50)

if st.button("Predict Resume Score"):

    # Create empty dataframe with same columns as training data
    input_df = pd.DataFrame(columns=X.columns)

    # Fill with zeros
    input_df.loc[0] = 0

    # Set only known numeric columns
    if "Experience_Years" in input_df.columns:
        input_df["Experience_Years"] = exp_input

    if "Technical_Score" in input_df.columns:
        input_df["Technical_Score"] = tech_input

    if "Soft_Skills_Score" in input_df.columns:
        input_df["Soft_Skills_Score"] = soft_input

    prediction = model.predict(input_df)

    st.success(f"Predicted Resume Score: {round(prediction[0],2)}")