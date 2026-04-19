from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pandas as pd

# load spaCy model
nlp = spacy.load("en_core_web_sm")

def calculate_tfidf(text):
    # tokenize and lemmatize using spaCy
    doc = nlp(text.lower())
    filtered_tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop
    ]
    preprocessed_text = " ".join(filtered_tokens)

    # compute tf-idf
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_text])

    # convert tf-idf matrix to df
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(),
                columns=tfidf_vectorizer.get_feature_names_out())

    return df_tfidf, tfidf_vectorizer.get_feature_names_out()

# application to text
text = "."
tfidf_df, vocabulary = calculate_tfidf(text)

# display result
print("TF-IDF DataFrame:")
print(tfidf_df)