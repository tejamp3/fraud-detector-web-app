import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

def build_and_evaluate_model():
    """
    Builds and evaluates a fraud detection model using a real-world Kaggle dataset.
    The model is then saved for future use.
    """
    print("Starting model building and evaluation...")
    try:
        # Step 1: Load the dataset
        data_path = 'data/spam.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The file '{data_path}' was not found.")

        # Load the dataset. We'll manually handle the column names to avoid parsing errors.
        # This is the key data cleaning fix for this specific dataset.
        print("Loading data from 'data/spam.csv'...")
        df = pd.read_csv(data_path, encoding='latin-1', header=None)
        
        # Manually select the first two columns, as the rest are often empty.
        # This resolves the 'NaN' issue.
        df = df.iloc[:, :2]
        df.columns = ['label', 'message']
        
        # Display the first few rows to confirm it loaded correctly
        print("\nDataset loaded successfully. Here's a preview:")
        print(df.head())

        # Step 2: Data Preprocessing
        # The labels 'ham' (legitimate) and 'spam' (fraud) need to be converted to a numerical format.
        # This is a critical step for machine learning models.
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        # Crucial fix: Drop any rows where the label is NaN.
        # This handles malformed rows that aren't 'ham' or 'spam'.
        df.dropna(subset=['label'], inplace=True)

        # Step 3: Split the data into training and testing sets
        X = df['message']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("\nData split into training and testing sets.")

        # Step 4: Feature Engineering with TF-IDF
        # We transform the text messages into a matrix of TF-IDF features.
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        print("Text data vectorized using TF-IDF.")
        
        # Step 5: Model Training
        # We'll use a Naive Bayes classifier, which is a good baseline model for text classification.
        model = MultinomialNB()
        print("\nTraining the Naive Bayes model...")
        model.fit(X_train_vec, y_train)
        print("Model training complete.")

        # Step 6: Model Evaluation
        # We make predictions and calculate performance metrics.
        predictions = model.predict(X_test_vec)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)

        print("\nModel Performance Metrics on the test set:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        
        # Step 7: Saving the trained model and vectorizer
        # We save the model and the vectorizer for future predictions.
        joblib.dump(model, 'fraud_model.joblib')
        joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
        print("\nModel and vectorizer saved to 'fraud_model.joblib' and 'tfidf_vectorizer.joblib'")

        # Example of how to use the saved model to make a new prediction
        print("\n--- Testing the Saved Model ---")
        loaded_model = joblib.load('fraud_model.joblib')
        loaded_vectorizer = joblib.load('tfidf_vectorizer.joblib')
        
        new_message = ["hi all."]
        
        new_message_vec = loaded_vectorizer.transform(new_message)
        prediction = loaded_model.predict(new_message_vec)
        
        result = "Fraudulent" if prediction[0] == 1 else "Legitimate"
        print(f"The message '{new_message[0]}' is predicted to be: {result}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have downloaded the dataset and placed it in a 'data' subfolder.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\nScript execution complete.")

if __name__ == '__main__':
    build_and_evaluate_model()
