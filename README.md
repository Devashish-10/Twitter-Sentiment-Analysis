# Twitter Sentiment Analysis

This project implements a machine learning model to classify Twitter sentiment as positive or negative using Support Vector Machine (SVM) with TF-IDF vectorization.

## Project Overview

The model is trained on a dataset of 1.6 million tweets and achieves approximately 79% accuracy in sentiment classification. It uses advanced text preprocessing techniques and feature engineering to effectively classify tweet sentiments.

## Features

- Text preprocessing with NLTK (stopword removal, lemmatization)
- TF-IDF vectorization with optimized parameters
- Feature selection using Chi-square test
- SVM classifier with class balancing
- Comprehensive model evaluation metrics

## Requirements

- Python 3.8+
- Required Python packages:
  ```
  numpy
  pandas
  nltk
  scikit-learn
  joblib
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Twitter-Sentiment-Analysis
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK resources:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

## Dataset

The model is trained on the "Sentiment140" dataset, which contains 1.6 million tweets labeled as positive (4) or negative (0). The dataset includes the following columns:
- sentiment: 0 (negative) or 4 (positive)
- id: tweet ID
- date: tweet timestamp
- query: search query
- user: username
- text: tweet content

## Data Pipeline

The data processing pipeline consists of the following stages:

1. **Data Loading and Initial Processing**:
   ```python
   # Load dataset
   df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)
   df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
   
   # Filter and map sentiment values
   df = df[df['sentiment'].isin([0, 4])]
   df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
   ```

2. **Text Preprocessing Pipeline**:
   ```python
   def clean_text(text):
       # Remove URLs
       text = re.sub(r"http\S+", "", text)
       # Remove usernames and hashtags
       text = re.sub(r"@\w+", "", text)
       text = re.sub(r"#\w+", "", text)
       # Remove special characters and numbers
       text = re.sub(r"[^\w\s]", "", text)
       text = re.sub(r"\d+", "", text)
       # Convert to lowercase
       text = text.lower()
       # Tokenize and lemmatize
       tokens = word_tokenize(text)
       tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
       return " ".join(tokens)
   
   # Apply preprocessing
   df["clean_text"] = df["text"].apply(clean_text)
   df = df[df["clean_text"].str.strip() != ""]
   ```

3. **Feature Engineering Pipeline**:
   ```python
   # TF-IDF Vectorization
   vectorizer = TfidfVectorizer(
       max_features=25000,
       ngram_range=(1, 2),
       min_df=5,
       max_df=0.9,
       sublinear_tf=True
   )
   
   # Feature Selection
   selector = SelectKBest(chi2, k=15000)
   ```

4. **Model Training Pipeline**:
   ```python
   # Split data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )
   
   # Transform training data
   X_train_tfidf = vectorizer.fit_transform(X_train)
   X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
   
   # Train model
   model = LinearSVC(class_weight='balanced', C=0.25)
   model.fit(X_train_selected, y_train)
   ```

5. **Prediction Pipeline**:
   ```python
   def predict_sentiment(text):
       # Clean text
       cleaned_text = clean_text(text)
       # Transform text
       tfidf = vectorizer.transform([cleaned_text])
       selected = selector.transform(tfidf)
       # Predict
       prediction = model.predict(selected)
       return "Positive" if prediction[0] == 1 else "Negative"
   ```

The pipeline ensures consistent preprocessing and feature engineering for both training and prediction, maintaining data integrity throughout the process.

## Model Architecture

1. **Text Preprocessing**:
   - URL removal
   - Username and hashtag removal
   - Special character and number removal
   - Lowercase conversion
   - Tokenization and lemmatization
   - Stopword removal

2. **Feature Engineering**:
   - TF-IDF vectorization (25,000 features)
   - N-gram range: (1,2)
   - Feature selection using Chi-square (15,000 features)

3. **Model**:
   - LinearSVC classifier
   - Class weight balancing
   - Regularization parameter (C=0.25)

## Performance Metrics

### Training Set
- Accuracy: 79.85%
- Precision: 0.80
- Recall: 0.80
- F1-score: 0.80

### Test Set
- Accuracy: 79.05%
- Precision: 0.79
- Recall: 0.79
- F1-score: 0.79

## Usage

1. Load the saved model:
   ```python
   import joblib
   
   model = joblib.load("svc_twitter_sentiment.pkl")
   vectorizer = joblib.load("tfidf_vectorizer.pkl")
   selector = joblib.load("feature_selector.pkl")
   ```

2. Preprocess and predict sentiment:
   ```python
   def predict_sentiment(text):
       # Clean text
       cleaned_text = clean_text(text)
       # Transform text
       tfidf = vectorizer.transform([cleaned_text])
       selected = selector.transform(tfidf)
       # Predict
       prediction = model.predict(selected)
       return "Positive" if prediction[0] == 1 else "Negative"
   ```

## Model Files

- `svc_twitter_sentiment.pkl`: Trained SVM model
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer
- `feature_selector.pkl`: Feature selector

## Future Improvements

- Implement cross-validation for better model evaluation
- Experiment with different kernel functions in SVM
- Try deep learning approaches (LSTM, BERT)
- Add more sophisticated text preprocessing techniques
- Implement real-time sentiment analysis

## License

This project is created for self Learning Process

