import streamlit as st
from predict import FakeNewsPredictor
import time

# Custom Streamlit layout and aesthetics configurations
st.set_page_config(
    page_title="AI Fake News Detection",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern premium feel
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stTextArea textarea {
        background-color: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
    }
    .stButton button {
        background-color: #238636;
        color: #ffffff;
        border: None;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        transition: 0.2s ease-in-out;
    }
    .stButton button:hover {
        background-color: #2ea043;
        transform: scale(1.02);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 1rem;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("📰 AI Fake News Detection System")
st.markdown("Use natural language processing and machine learning to analyze news articles and determine their authenticity.")

# Initialize the predictor
@st.cache_resource
def load_predictor():
    return FakeNewsPredictor()

try:
    predictor = load_predictor()
    model_ready = predictor.is_model_loaded()
except Exception as e:
    model_ready = False
    st.error(f"Error loading model: {e}")

if not model_ready:
    st.warning("⚠️ Model is not trained yet. Please run `python train_model.py` to train the model first.")
else:
    # User Input Section
    news_text = st.text_area(
        "Enter the news article text below:",
        height=250,
        placeholder="Paste a news article here to analyze..."
    )

    if st.button("Predict Authenticity", use_container_width=True):
        if not news_text.strip():
            st.error("Please enter some text to analyze.")
        else:
            with st.spinner('Analyzing patterns with AI...'):
                time.sleep(0.5) # Simulate slight processing time for UX micro-animation effect
                try:
                    result = predictor.predict(news_text)
                    prediction = result['prediction']
                    confidence = result['confidence']
                    
                    if prediction.lower() == 'fake':
                        st.markdown(f"""
                        <div class="prediction-box" style="background-color: #3b1418; border: 1px solid #da3633;">
                            <h2 style="color: #ff7b72; margin-bottom: 0;">🛑 FAKE NEWS DETECTED</h2>
                            <p style="font-size: 1.2rem; color: #ff7b72;">Confidence Score: <b>{confidence:.1f}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box" style="background-color: #172b1d; border: 1px solid #238636;">
                            <h2 style="color: #56d364; margin-bottom: 0;">✅ REAL NEWS</h2>
                            <p style="font-size: 1.2rem; color: #56d364;">Confidence Score: <b>{confidence:.1f}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.success("Analysis complete!")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #8b949e; font-size: 0.9em;'>Built with Streamlit & Scikit-learn</p>", unsafe_allow_html=True)
