import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st


st.title("MOVIE GENRE PREDICTOR SYSTEM")
# Input field for movie title or description
abc = st.text_input("Enter Movie Title or Description")
zoner_Description = [abc]
if st.button("Predict Genre"):

    data = pd.read_csv("E:\MovieGenreClassification\description.txt") #add the file path

    def load_data(file_path):
        with open(file_path, 'r', encoding='utf') as f:
            data = f.readlines()
        data = [line.strip().split(' ::: ') for line in data]
        return data


    train_data = load_data("E:\MovieGenreClassification\\train_data.txt")
    train_df = pd.DataFrame(train_data, columns=['ID', 'Title', 'Genre', 'Description'])

    test_data = load_data("E:\MovieGenreClassification\\test_data.txt")
    test_df = pd.DataFrame(test_data, columns=['ID', 'Title', 'Description'])

    test_solution = load_data("E:\MovieGenreClassification\\test_data_solution.txt")
    test_solution_df = pd.DataFrame(test_solution, columns=['ID', 'Title', 'Genre', 'Description'])

    # feature extraction
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2), max_df=0.85, min_df=2)

    X_train_tfidf = vectorizer.fit_transform(train_df['Title'] + " " + train_df['Description'])
    X_test_tfidf = vectorizer.transform(test_df['Title'] + " " + test_df['Description'])

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['Genre'])

    # model - logistic regression
    lr_model = LogisticRegression(C=10, max_iter=1000, class_weight='balanced', solver='liblinear')
    lr_model.fit(X_train_tfidf, y_train)

    y_pred = lr_model.predict(X_test_tfidf)
    predicted_genres = label_encoder.inverse_transform(y_pred)

    test_df['Predicted_Genre'] = predicted_genres
    merged_df = pd.merge(test_solution_df[['ID', 'Genre']], test_df[['ID', 'Predicted_Genre']], on='ID')

    accuracy = accuracy_score(merged_df['Genre'], merged_df['Predicted_Genre'])
    acc= accuracy*100
    # classificationReport = classification_report(merged_df['Genre'], merged_df['Predicted_Genre'])

    # testing

    test_data_tfidf = vectorizer.transform(zoner_Description)
    y_pred_lr = lr_model.predict(test_data_tfidf)
    predicted_genres_lr = label_encoder.inverse_transform(y_pred_lr)

    st.write(f"Accuracy: {acc} %")
    st.write(f"Predicted Genre: {predicted_genres_lr}")
