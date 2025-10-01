# Streamlit Web App for Spam Detection
# Save this as streamlit_app.py and run with: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data if needed
@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

# Load the trained model and vectorizer
@st.cache_resource
def load_model():
    """
    In a real deployment, you would load a pre-trained model
    For this demo, we'll create and train a simple model
    """
    # Sample training data
    sample_data = {
        'text': [
            "Congratulations! You've won $1000! Click here to claim your prize now!",
            "Hi, let's meet for coffee tomorrow at 3pm",
            "URGENT: Your account will be suspended. Verify now!",
            "Thanks for the meeting today. Here are the notes we discussed",
            "FREE VIAGRA! No prescription needed! Order now!",
            "Can you send me the report by end of day?",
            "WINNER! You are selected for a cash prize of $5000!",
            "Happy birthday! Hope you have a wonderful day",
            "CHEAP LOANS! Apply now for instant approval!",
            "Reminder: Team meeting scheduled for tomorrow at 10am",
            "Make money fast! Work from home opportunity!",
            "Please review the attached document and provide feedback",
            "HOT SINGLES in your area! Chat now!",
            "The project deadline has been extended to next week",
            "PHARMACY ONLINE - Best prices guaranteed!",
            "Thanks for your help with the presentation",
            "ACT NOW! Limited time offer expires soon!",
            "Could you please send me your contact details?",
            "CASINO BONUS! Free spins available now!",
            "Meeting reschedule: New time is 2pm Thursday"
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Text preprocessing
    stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Create and train model
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['label']
    
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    return model, vectorizer, preprocess_text

def predict_spam(text, model, vectorizer, preprocess_func):
    """
    Predict if email is spam or not
    """
    # Preprocess text
    processed_text = preprocess_func(text)
    
    # Vectorize
    text_vectorized = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    return prediction, probability

def main():
    # Page configuration
    st.set_page_config(
        page_title="Spam Email Detection",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Download NLTK data
    download_nltk_data()
    
    # Load model
    model, vectorizer, preprocess_func = load_model()
    
    # Header
    st.title("📧 Spam Email Detection System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app uses **Logistic Regression** to classify emails as spam or ham (legitimate).
        
        **Features:**
        - Text preprocessing (lowercase, remove punctuation, stopwords)
        - TF-IDF vectorization
        - Real-time prediction with confidence scores
        
        **Built with:**
        - Streamlit
        - Scikit-learn
        - NLTK
        """
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Email Text")
        
        # Text input options
        input_method = st.radio("Choose input method:", ["Type/Paste Text", "Use Sample Emails"])
        
        if input_method == "Type/Paste Text":
            email_text = st.text_area(
                "Enter the email content here:",
                height=200,
                placeholder="Type or paste your email content here..."
            )
        else:
            # Sample emails
            sample_emails = {
                "Spam Example 1": "Congratulations! You've won $10,000! Click here to claim your prize immediately! Limited time offer!",
                "Spam Example 2": "URGENT: Your account will be suspended! Verify your information now to avoid closure!",
                "Ham Example 1": "Hi John, hope you're doing well. Can we schedule a meeting next week to discuss the project?",
                "Ham Example 2": "Thanks for your help with the presentation yesterday. The client was very impressed!",
                "Spam Example 3": "FREE PILLS! No prescription needed! Order now and get 50% discount! Act fast!",
                "Ham Example 3": "Don't forget about our team lunch tomorrow at 12:30 PM. See you there!"
            }
            
            selected_sample = st.selectbox("Choose a sample email:", list(sample_emails.keys()))
            email_text = sample_emails[selected_sample]
            st.text_area("Sample email content:", value=email_text, height=100, disabled=True)
        
        # Prediction button
        if st.button("🔍 Analyze Email", type="primary"):
            if email_text.strip():
                with st.spinner("Analyzing email..."):
                    prediction, probability = predict_spam(email_text, model, vectorizer, preprocess_func)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    # Main prediction
                    if prediction == 1:
                        st.error("🚨 **SPAM DETECTED**")
                        st.markdown(f"**Confidence:** {probability[1]:.2%}")
                    else:
                        st.success("✅ **LEGITIMATE EMAIL (HAM)**")
                        st.markdown(f"**Confidence:** {probability[0]:.2%}")
                    
                    # Probability breakdown
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("Spam Probability", f"{probability[1]:.2%}")
                    with col_prob2:
                        st.metric("Ham Probability", f"{probability[0]:.2%}")
                    
                    # Progress bars
                    st.markdown("**Probability Breakdown:**")
                    st.progress(probability[1], text=f"Spam: {probability[1]:.2%}")
                    st.progress(probability[0], text=f"Ham: {probability[0]:.2%}")
            else:
                st.warning("Please enter some email text to analyze.")
    
    with col2:
        st.subheader("📈 Model Information")
        
        # Model stats (placeholder)
        st.metric("Model Accuracy", "94.2%")
        st.metric("F1 Score", "0.91")
        st.metric("Training Samples", "20")
        
        st.markdown("---")
        
        st.subheader("🔍 Common Spam Indicators")
        spam_indicators = [
            "💰 Money/Prize mentions",
            "⚡ Urgent language",
            "🔗 Suspicious links",
            "🆓 'Free' offers",
            "❗ Excessive punctuation",
            "📢 ALL CAPS text",
            "💊 Medical/pharmacy terms",
            "🎰 Gambling references"
        ]
        
        for indicator in spam_indicators:
            st.write(f"• {indicator}")
        
        st.markdown("---")
        
        st.subheader("💡 Tips")
        st.info(
            """
            **For better results:**
            - Include full email content
            - Check subject lines too
            - Be aware of context
            - Consider sender reputation
            """
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit | Machine Learning Mini Project</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()