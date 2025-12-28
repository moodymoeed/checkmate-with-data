import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from PIL import Image
import os

# I am setting the page config to wide mode for better visualization
st.set_page_config(page_title="Checkmate with Data", layout="wide", page_icon="♟️")

# --- LOADING RESOURCES ---
@st.cache_resource
def load_resources():
    # I'm loading the model and column list once to save time
    model = joblib.load('models/chess_model.pkl')
    model_columns = joblib.load('models/model_columns.pkl')
    return model, model_columns

@st.cache_data
def load_data():
    # Loading the raw games for the story section
    df = pd.read_csv('data/chess_games_raw.csv')
    return df

# --- PAGE STRUCTURE ---
page = st.sidebar.radio("Navigate", ["Project Story & Insights", "Win Predictor"])

# --- PAGE 1: PROJECT STORY & INSIGHTS ---
if page == "Project Story & Insights":
    st.title("♟️ Checkmate with Data: My Summer Chess Journey")
    st.markdown("""
    Welcome to my data science portfolio! This project analyzes my personal chess improvement over Summer 2025.
    I tracked every move, every win, and every blunder to find out what actually moved the needle.
    """)

    # 1. Rating Trend
    st.header("1. The Climb 📈")
    st.write("First, let's look at my rating over time. Did I actually get better?")
    try:
        image = Image.open('images/viz_1_5_rating_trend.png')
        st.image(image, caption='My Rating Evolution', use_container_width=True)
    except FileNotFoundError:
        st.error("Image not found: images/viz_1_5_rating_trend.png")

    # 2. Habits (When do I play?)
    st.header("2. Playing Habits 🕰️")
    st.write("Does playing late at night hurt my performance? Here is a heatmap of my games by Day and Hour.")
    try:
        img_habits = Image.open('images/viz_5_5_habits.png')
        st.image(img_habits, use_container_width=True)
    except FileNotFoundError:
        st.write("Habits chart missing.")

    # 3. Interactive Sunburst (The "Wow" Factor)
    st.header("3. Opening Repertoire 🧩")
    st.write("Explore my opening choices! Click on the inner rings to zoom in.")
    
    df = load_data()
    
    # Simple hierarchy extraction logic
    def extract_hierarchy(opening_str):
        parts = opening_str.split(':') # Most PGNs use colon sep
        if len(parts) > 1:
            return parts[0].strip(), parts[1].strip()
        parts = opening_str.split(' ') # Fallback
        return parts[0], " ".join(parts[1:]) if len(parts) > 1 else "Main Line"

    df[['Family', 'Variation']] = df['opening'].apply(lambda x: pd.Series(extract_hierarchy(str(x))))
    
    # Filter for top openings to keep chart readable
    top_openings = df['Family'].value_counts().nlargest(15).index
    df_sunburst = df[df['Family'].isin(top_openings)]
    
    fig = px.sunburst(
        df_sunburst, 
        path=['Family', 'Variation'], 
        title="My Most Frequent Openings (Interactive)",
        width=800, height=800
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. Key Insights (Merged Hypothesis Section)
    st.header("4. Key Insights 💡")
    st.markdown("Here are the two biggest takeaways from my statistical analysis.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Insight A: The White Advantage")
        st.write("Playing White gives a statistically significant advantage due to the first-move initiative.")
        try:
            img_hyp1 = Image.open('images/viz_3_5_color.png')
            st.image(img_hyp1, caption='Win Rate by Color', use_container_width=True)
        except:
            st.error("Missing Color chart")

    with col2:
        st.subheader("Insight B: Game Length")
        st.write("Longer games favor my opponents. Shorter games (quick tactics) are my strong suit.")
        try:
            # Using the hypothesis chart that shows game moves vs outcome
            img_moves = Image.open('images/viz_hypothesis_2.png')
            st.image(img_moves, caption='Game Length impact on Results', use_container_width=True)
        except:
            st.error("Missing Game Length chart")

    # 5. Raw Data (Credibility)
    with st.expander("👀 Peek at the Raw Data"):
        st.write("Here is a sample of the actual game logs I analyzed:")
        st.dataframe(df.head())


# --- PAGE 2: WIN PREDICTOR ---
elif page == "Win Predictor":
    st.title("🔮 Chess Win Predictor")
    st.markdown("""
    I trained a Machine Learning model (Logistic Regression) to predict my win probability based on game state.
    **Adjust the sliders sidebar to see if I win this position!**
    """)

    model, model_columns = load_resources()

    # --- SIDEBAR INPUTS ---
    st.sidebar.header("Game State Inputs")
    
    material = st.sidebar.number_input("Material Difference", min_value=-10, max_value=10, value=0, help="Positive = I have more material")
    mobility = st.sidebar.slider("Mobility Score", 0, 60, 30, help="Number of legal moves available")
    color_input = st.sidebar.radio("My Color", ["White", "Black"])
    king_moved = st.sidebar.checkbox("Has King Moved?", value=False, help="Check if the King has moved")
    
    openings = [c.replace('open_', '') for c in model_columns if c.startswith('open_')]
    opening_choice = st.sidebar.selectbox("Opening", ['Other'] + openings)

    # --- PREDICTION LOGIC ---
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0 # Fill with zeros

    input_data['material_diff'] = material
    input_data['mobility_count'] = mobility
    input_data['king_moved'] = 1 if king_moved else 0
    input_data['is_white'] = 1 if color_input == "White" else 0
    
    if opening_choice != 'Other':
        col_name = f"open_{opening_choice}"
        if col_name in input_data.columns:
            input_data[col_name] = 1

    prob = model.predict_proba(input_data)[0][1] # Probability of Class 1 (Win)
    
    # --- DISPLAY RESULTS ---
    col_main, col_viz = st.columns([1, 2])

    with col_main:
        st.subheader("Prediction")
        
        delta_color = "normal"
        if prob > 0.6: delta_color = "off"
        elif prob < 0.4: delta_color = "inverse"
        
        st.metric(label="Win Probability", value=f"{prob:.1%}", delta=f"{prob-0.5:.1%} vs Coin Flip", delta_color=delta_color)

        if 0.2 < prob < 0.5:
            st.warning("⚠️ The stats say you are losing, but the **Optimist Model** says keep fighting!")
        elif prob >= 0.5:
            st.success("🎉 Looking good! Don't blunder!")
        else:
            st.error("💀 It's not looking good chief...")

    with col_viz:
        st.subheader("Why this prediction?")
        st.write("Here are the features that matter most to the model:")
        try:
            img_coeff = Image.open('images/ml_1_coefficients.png')
            st.image(img_coeff, use_container_width=True)
        except:
            st.write("Feature chart missing")

    # --- TECHNICAL EXPANDER ---
    with st.expander("🤓 Technical Details (Model Performance)"):
        st.write("Model Benchmarks & Threshold Analysis:")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image('images/ml_2_model_comparison.png', caption="Model Comparison", use_container_width=True)
        with c2:
            st.image('images/ml_4_confusion_matrix.png', caption="Confusion Matrix", use_container_width=True)
        with c3:
            try:
                st.image('images/ml_3_threshold.png', caption="Threshold Tuning", use_container_width=True)
            except:
                st.write("Threshold chart missing")
