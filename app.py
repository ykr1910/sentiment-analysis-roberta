import streamlit as st
from sentiment_analyzer import analyze_roberta_label

# Page configuration
st.set_page_config(page_title="Sentiment Analysis", page_icon="📝", layout="centered")

# Title and description
st.title("📝 Sentiment Analysis Web App")
st.markdown("""
Enter your comment in the box below, and click **Analyze** to detect whether it's **Positive**, **Neutral**, or **Negative** sentiment.
""")

# Text input with height adjustment
text_input = st.text_area("Enter your comment here:", height=150)

# Analyze button centered
if st.button("Analyze"):
    if text_input.strip():
        result = analyze_roberta_label(text_input)
        
        # Colored sentiment display
        sentiment = result['label']
        score = result['score']
        
        if sentiment == "POSITIVE":
            st.success(f"Sentiment: {sentiment} 😊")
        elif sentiment == "NEGATIVE":
            st.error(f"Sentiment: {sentiment} 😠")
        else:
            st.info(f"Sentiment: {sentiment} 😐")
        
        st.write(f"Confidence: {score:.3f}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer with small text
st.markdown("---")
st.markdown("*Built with Streamlit & RoBERTa Sentiment Model*")
