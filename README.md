# 🎬 IMDb Movie Review Sentiment Analysis

A machine learning project that classifies IMDb movie reviews as **Positive** or **Negative** using Natural Language Processing and machine learning.

## 📌 Project Overview

The objective of this project is to develop a sentiment analysis system that automatically identifies the sentiment expressed in movie reviews.

The project follows a complete machine learning workflow:

**Data → Preprocessing → TF-IDF → Model Training → Evaluation → Model Saving → Streamlit Deployment**

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* NLTK
* TF-IDF
* Logistic Regression
* Support Vector Machine (SVM)
* Joblib
* Streamlit
* JupyterLab

## 📂 Project Structure

```text
Sentiment_Analysis/
├── app/
│   └── app.py
├── dataset/
├── models/
│   ├── svm_sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── results/
├── IMDb_Sentiment_Analysis.ipynb
├── README.md
└── .gitignore
```

## 🔄 Methodology

### 1. Data Collection

The IMDb movie review dataset was used for sentiment classification.

### 2. Text Preprocessing

The reviews were processed using:

* Lowercase conversion
* HTML tag removal
* Special character removal
* Whitespace normalization
* Stop-word removal
* Lemmatization

### 3. Feature Extraction

TF-IDF was used to convert the cleaned reviews into numerical feature vectors.

The implementation uses both:

* Unigrams
* Bigrams

### 4. Model Training

Two classification models were trained:

* Logistic Regression
* Linear Support Vector Machine (SVM)

### 5. Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### 6. Deployment

The trained SVM model and TF-IDF vectorizer were saved using Joblib and integrated into a Streamlit web application.

## 🚀 Run the Project Locally

Clone the repository:

```bash
git clone https://github.com/Shivam3831143/IMDb-Sentiment-Analysis.git
```

Move into the project directory:

```bash
cd IMDb-Sentiment-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```

## 📊 Results

The performance of Logistic Regression and SVM was compared using Accuracy, Precision, Recall and F1-Score.

The final model was selected based on the actual evaluation results obtained during experimentation.

## 💡 Application

The application accepts a movie review from the user and predicts whether the review expresses:

* 😊 Positive Sentiment
* 😞 Negative Sentiment

## 🔮 Future Improvements

Possible improvements include:

* Using Transformer-based models such as BERT
* Improving handling of negation and context
* Adding probability/confidence information
* Supporting more sentiment categories
* Deploying the application as a scalable web service
