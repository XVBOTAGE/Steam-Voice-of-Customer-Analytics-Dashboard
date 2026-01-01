import streamlit as st
import pandas as pd
from textblob import TextBlob
from collections import Counter
import re
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Steam Review Analysis Dashboard",
    page_icon="🎮",
    layout="wide"
)

# ==========================================
# 2. Helper Functions (Core Logic)
# ==========================================

@st.cache_data
def load_data(file_path, sample_size):
    """
    Load a sample of the dataset for performance optimization.
    
    Args:
        file_path (str): Path to the CSV file.
        sample_size (int): Number of rows to load to prevent memory overflow.
    
    Returns:
        pd.DataFrame: The processed dataframe.
        str: The name of the column containing review text.
    """
    if not os.path.exists(file_path):
        return None, None
    
    try:
        # Attempt to read with standard UTF-8 encoding
        df = pd.read_csv(file_path, nrows=sample_size, encoding='utf-8')
    except UnicodeDecodeError:
        # Fallback to ISO-8859-1 for legacy datasets
        df = pd.read_csv(file_path, nrows=sample_size, encoding='ISO-8859-1')
    
    # Intelligent Column Detection: Automatically find the review text column
    possible_names = ['review_text', 'review', 'text', 'body', 'comment', 'user_review']
    target_col = next((col for col in df.columns if col.lower() in possible_names), None)
    
    if target_col:
        # Data Cleaning: Remove empty rows
        df = df.dropna(subset=[target_col])
        return df, target_col
    
    return None, None

def analyze_sentiment(text):
    """
    Perform Sentiment Analysis using NLP.
    Returns a score between -1.0 (Negative) and 1.0 (Positive).
    """
    if pd.isna(text) or text == '':
        return 0, "Neutral"
    
    # Convert to string and lower case for consistency
    text_str = str(text).lower()
    score = TextBlob(text_str).sentiment.polarity
    
    # Define thresholds for business logic
    if score > 0.1: 
        return score, "Positive"
    elif score < -0.1: 
        return score, "Negative"
    else: 
        return score, "Neutral"

def get_keywords(text_series):
    """
    Extract top keywords excluding common stopwords.
    """
    # Combine all text into one large corpus
    all_text = " ".join(text_series.astype(str)).lower()
    # Regex to find words of length 3-15
    words = re.findall(r'\b[a-z]{3,15}\b', all_text)
    
    # Business-specific Stopwords (words to ignore)
    stopwords = set([
        'the', 'and', 'game', 'play', 'this', 'that', 'for', 'with', 'you', 'not', 
        'was', 'but', 'are', 'have', 'just', 'like', 'really', 'can', 'from', 'out',
        'playing', 'played', 'get', 'all', 'time', 'would'
    ])
    
    clean_words = [w for w in words if w not in stopwords]
    return Counter(clean_words).most_common(10)

# ==========================================
# 3. Main Dashboard UI
# ==========================================

# Sidebar: Configuration Controls
st.sidebar.title("🔧 Configuration")

# Default path set to your specific local file
DEFAULT_PATH = "/Users/elliotwho/PERSONAL FILES/WORK/dataset.csv"
file_path = st.sidebar.text_input("CSV File Path", value=DEFAULT_PATH)

# Explained: Why we use a slider? To handle Big Data performance issues in a demo environment.
sample_size = st.sidebar.slider("Sample Size (Rows)", min_value=1000, max_value=50000, value=5000, step=1000)
st.sidebar.info("Note: Processing 6.4M rows requires offline batch processing (e.g., Spark). We use sampling for this real-time demo.")

run_button = st.sidebar.button("Run Analysis")

if run_button:
    st.session_state['run'] = True

# Main Page Header
st.title("🎮 Steam Review AI Analytics")
st.markdown("### Voice of Customer (VoC) Dashboard")
st.markdown("This dashboard utilizes NLP to transform unstructured customer feedback into actionable insights.")

# Dashboard Logic
if 'run' in st.session_state and st.session_state['run']:
    
    with st.spinner('Loading data and running AI models... (This may take a moment)'):
        df, target_col = load_data(file_path, sample_size)
        
        if df is not None and target_col is not None:
            # Execute Sentiment Analysis pipeline
            df['score'], df['label'] = zip(*df[target_col].apply(analyze_sentiment))
            
            # --- KPI Section ---
            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("Reviews Analyzed", f"{len(df):,}") # Format with comma
            
            pos_pct = (df['label'] == 'Positive').mean() * 100
            col2.metric("Positive Sentiment Rate", f"{pos_pct:.1f}%")
            
            neg_count = (df['label'] == 'Negative').sum()
            col3.metric("Negative Alerts", neg_count, delta_color="inverse")
            
            # --- Visualization Section ---
            st.divider()
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Sentiment Distribution")
                sentiment_counts = df['label'].value_counts()
                
                # Matplotlib visualization
                fig1, ax1 = plt.subplots()
                colors = {'Positive':'#66b3ff', 'Neutral':'#99ff99', 'Negative':'#ff9999'}
                ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
                        colors=[colors.get(x, '#cccccc') for x in sentiment_counts.index],
                        startangle=90)
                st.pyplot(fig1)
                
            with c2:
                st.subheader("Top Keywords")
                keywords = get_keywords(df[target_col])
                if keywords:
                    words, counts = zip(*keywords)
                    fig2, ax2 = plt.subplots()
                    ax2.barh(words, counts, color='skyblue')
                    ax2.invert_yaxis() 
                    st.pyplot(fig2)
                else:
                    st.warning("Not enough data to extract keywords.")

            # --- Live Demo Section ---
            st.divider()
            st.subheader("🤖 Live AI Tester")
            st.caption("Test the model with your own text to verify accuracy.")
            
            user_input = st.text_area("Enter a mock review:", "Great graphics but the server crashes all the time!")
            
            if user_input:
                score, label = analyze_sentiment(user_input)
                st.write(f"**AI Prediction:** {label} (Score: {score:.2f})")
                
                # Dynamic UI feedback based on result
                if label == "Negative":
                    st.error("⚠️ Negative Feedback Detected! High Priority.")
                elif label == "Positive":
                    st.success("✅ Positive Feedback. Low Priority.")
                else:
                    st.info("ℹ️ Neutral Feedback.")
            
            # --- Data Inspection ---
            with st.expander("📂 View Raw Data Sample"):
                st.dataframe(df[[target_col, 'label', 'score']].head(100))
                
        else:
            if df is None:
                st.error(f"File not found: {file_path}. Please check the path.")
            elif target_col is None:
                st.error("Could not automatically detect the review column. Please check your CSV headers.")
else:
    st.info("👈 Please configure the settings in the sidebar and click 'Run Analysis' to start.")