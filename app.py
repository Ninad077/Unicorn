import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load Q&A data
def load_data():
    return pd.read_csv('qna.csv', encoding='latin1', delimiter=';')

def get_most_similar_question(new_sentence, vectorizer, tfidf_matrix, questions, answers):
    new_tfidf = vectorizer.transform([new_sentence])

    similarities = cosine_similarity(new_tfidf, tfidf_matrix)

    most_similar_index = np.argmax(similarities)

    similarity_percentage = similarities[0, most_similar_index] * 100

    return answers[most_similar_index], similarity_percentage

def answer_the_question(new_sentence, vectorizer, tfidf_matrix, questions, answers):
    most_similar_answer, similarity_percentage = get_most_similar_question(new_sentence, vectorizer, tfidf_matrix, questions, answers)
    if similarity_percentage > 70:
        return most_similar_answer
    else:
        return 'Sorry, I am not aware of this information :('

def main():
    
    st.markdown(
        "<h1 style='text-align: center; color: #269A96;'>Unicorn</h1>",
        unsafe_allow_html=True
    )

    # Load Q&A data
    data = load_data()
    questions = data['question'].tolist()
    answers = data['answer'].tolist()

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)

    # Get user input
    user_question = st.text_input("Ask me a question:", key="user_question")

    # Apply CSS to style the text input field
    st.markdown(
        """
        <style>
        .st-cc {
            background-color: #269A96 !important;
            color: #FFFFFF !important;
        }
        .st-bk {
            color: #FFFFFF !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Process the question when submitted
    if st.button("Submit", key="submit_button", help="Submit your question"):
        if user_question:
            response = answer_the_question(user_question, vectorizer, tfidf_matrix, questions, answers)
            st.write(response)
        else:
            st.warning("Please ask a question.")

    # Apply CSS to style the button
    st.markdown(
        """
        <style>
        .css-1bkz6v6 {
            background-color: #a5eae9;
            color: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
