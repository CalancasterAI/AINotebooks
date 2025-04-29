import tensorflow as tf
import streamlit as st
import joblib
import io
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.sequence import pad_sequences
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(page_title="Tweet Sentiment Analyzer", layout="wide")

st.markdown("""
    <style>
        .block-container {
            max-width: 1200px;
            padding: 2rem 2rem;
            margin: auto;
            background-color: rgb(8, 10, 16);
            border-radius: 30px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

def confidence_bar(conf):
    filled = int(conf // 5)
    empty = 20 - filled
    return "🟩" * filled + "⬜" * empty + f" ({round(conf, 1)}%)"

model = joblib.load('models/sentiment_model2.pkl')

with open('models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_SEQUENCE_LENGTH = 30

st.title("Tweet Sentiment Analyzer")

user_input = st.text_area("Enter a tweet for sentiment analysis: ")

if st.button("Analyze"):
    if user_input.strip() != "":
        # Tokenize & Pad
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

        # Predict
        prediction = model.predict(padded_sequence)[0][0]

        # Calculate confidence
        confidence = prediction if prediction >= 0.5 else (1 - prediction)

        st.write(f"### Prediction: {'Positive 😀' if prediction >= 0.5 else 'Negative 😠'}")
        
        # This will show a standard loading-style green progress bar
        st.progress(float(confidence))

        # Also display numeric confidence
        st.write(f"Confidence: {round(confidence*100, 2)}%")
    else:
        st.warning("Please enter a tweet!")

st.write("---")
st.write("Or upload a CSV file of tweets to analyze:")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        if 'text' not in data.columns:
            st.error("Uploaded CSV must have a 'text' column.")

        else:
            tweets = data['text']

           # Tokenize + Pad
            sequences = tokenizer.texts_to_sequences(tweets)
            padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

            # Predict
            predictions = model.predict(padded_sequences).flatten()

            # Attach predictions back to dataframe
            data['Prediction'] = ['Positive 😀' if pred >= 0.5 else 'Negative 😠' for pred in predictions]
            data['Confidence'] = [pred if pred >= 0.5 else (1-pred) for pred in predictions]
            data['Confidence (%)'] = [round(pred*100, 2) if pred >= 0.5 else round((1-pred)*100, 2) for pred in predictions]

            st.success("Predictions completed successfully!")
            
            data['Confidence Bar'] = data['Confidence (%)'].apply(confidence_bar)
            
            csv = data.to_csv(index=False).encode('utf-8')

            st.info("You can download your predictions as a CSV below:")
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name='tweet_predictions.csv',
                mime='text/csv',
            )

            st.write("### Prediction Results:")

            st.dataframe(data[['text', 'Prediction', 'Confidence Bar']])

            # --- Highlight low-confidence predictions ---
            low_confidence_threshold = 60  # adjust if you want
            low_confidence = data[data['Confidence (%)'] < low_confidence_threshold]

            if not low_confidence.empty:
                st.write("---")
                st.warning(f"### Tweets with Confidence Below {low_confidence_threshold}%:")
                st.dataframe(low_confidence[['text', 'Prediction', 'Confidence (%)']])
            else:
                st.info("All predictions have confidence above threshold.")

            st.write("---")
            st.write("### Word Clouds")

            # Separate positive and negative tweets
            positive_tweets = " ".join(data[data['Prediction'] == 'Positive 😀']['text'])
            negative_tweets = " ".join(data[data['Prediction'] == 'Negative 😠']['text'])

            # Plot WordClouds
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Positive Tweets WordCloud")
                if positive_tweets:
                    wordcloud_pos = WordCloud(width=400, height=300, background_color='white').generate(positive_tweets)
                    plt.imshow(wordcloud_pos, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt)
                else:
                    st.write("No positive tweets.")

            with col2:
                st.subheader("Negative Tweets WordCloud")
                if negative_tweets:
                    wordcloud_neg = WordCloud(width=400, height=300, background_color='white').generate(negative_tweets)
                    plt.imshow(wordcloud_neg, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt)
                else:
                    st.write("No negative tweets.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    else:
        st.error("Uploaded CSV must have a 'text' column.")
