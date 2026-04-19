import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── LOAD MODEL (runs once, cached for speed) ──────────────

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, le, features
# WHY @st.cache_resource?
# Without it, Streamlit reloads the model on EVERY user interaction.
# cache_resource loads it once and keeps it in memory. Much faster.

model, le, feature_names = load_model()

# ── PAGE LAYOUT ───────────────────────────────────────────

st.title("🎯 Focus Score Predictor")
st.markdown("Enter your daily habits below and get a predicted focus score.")
st.divider()


col1, col2 = st.columns(2)
# WHY two columns? Better layout — doesn't force the user to scroll.

with col1:
    sleep_hours = st.slider("🛌 Sleep hours",    min_value=4,  max_value=9,  value=7)
    screen_time = st.slider("📱 Screen time (hrs)", min_value=1, max_value=8, value=3)
    study_hours = st.slider("📚 Study hours",    min_value=1,  max_value=8,  value=4)

with col2:
    breaks      = st.slider("☕ Breaks taken",   min_value=0,  max_value=5,  value=2)
    noise_level = st.selectbox("🔊 Noise level",   options=['Low', 'Medium', 'High'])
    # WHY selectbox? noise_level is a category, not a number.
    # selectbox gives a dropdown — easier than typing.

st.divider()

# ── PREDICT BUTTON ─────────────────────────────────────────

if st.button("🔍 Predict My Focus Score", type="primary"):

    # Encode noise_level exactly like we did in training
    noise_encoded = le.transform([noise_level])[0]
    # WHY [0]? le.transform returns an array. [0] gets the first (only) element.

    # Build the input row — MUST be in the same column order as training
    input_data = pd.DataFrame([[sleep_hours, screen_time, study_hours,
                                 breaks, noise_encoded]],
                               columns=feature_names)

    # Make prediction
    predicted_score = model.predict(input_data)[0]
    predicted_score = int(np.clip(predicted_score, 0, 100))


    # ── DISPLAY THE SCORE ──────────────────────────────────

    if predicted_score >= 70:
        color = "🟢"
        label = "Excellent focus day!"
    elif predicted_score >= 50:
        color = "🟡"
        label = "Moderate focus — some room to improve"
    else:
        color = "🔴"
        label = "Low focus — check the suggestions below"

    st.metric(
        label  = f"{color} Predicted Focus Score",
        value  = f"{predicted_score} / 100",
        delta  = label
    )

    # ── INSIGHTS: WHY IS IT THIS SCORE? ───────────────────

    st.subheader("💡 What's affecting your score?")

    insights = []

    if sleep_hours < 6:
        insights.append(f"😴 Sleep is low ({sleep_hours}h). Aim for 7–8h. Impact: +{(7-sleep_hours)*5} pts")
    if screen_time > 5:
        insights.append(f"📱 High screen time ({screen_time}h). Try cutting to 3h. Impact: +{(screen_time-3)*4} pts")
    if study_hours < 3:
        insights.append(f"📚 Low study hours ({study_hours}h). Aim for 4–5h. Impact: +{(4-study_hours)*4} pts")
    if breaks == 0:
        insights.append("☕ No breaks taken. Short breaks improve focus significantly.")
    if noise_level == 'High':
        insights.append("🔊 High noise environment reduces focus by ~15 points. Try headphones or a quiet space.")
    if noise_level == 'Medium':
        insights.append("🔊 Medium noise reduces focus by ~7 points. A quieter space would help.")

    if insights:
        for tip in insights:
            st.info(tip)
    else:
        st.success("✅ All habits look great! Keep it up.")


      # ── FEATURE IMPORTANCE CHART ──────────────────────────

    st.subheader("📊 What matters most for focus?")
    importance_df = pd.DataFrame({
        'Factor'     : feature_names,
        'Importance' : model.feature_importances_
    }).sort_values('Importance', ascending=False)

    st.bar_chart(importance_df.set_index('Factor'))
    # WHY st.bar_chart? It's the simplest way to show a chart in Streamlit.
    # No matplotlib setup needed.