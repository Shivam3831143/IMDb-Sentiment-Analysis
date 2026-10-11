# IMDb Sentiment Analysis Using Machine Learning

## 1. Project Overview

The **IMDb Sentiment Analysis** project uses Machine Learning and Natural Language Processing (NLP) to identify whether a movie review expresses a positive or negative opinion.

People share their opinions about movies through online reviews. Reading thousands of reviews manually can be time-consuming. This project helps automate the process by analyzing the text of a review and predicting its sentiment.

The project uses IMDb movie reviews to train and evaluate a machine learning model. A Support Vector Machine (SVM) classifier is used to predict whether a review is positive or negative. A Streamlit web application provides a simple interface where users can enter a review and see the predicted sentiment.

## 2. Problem Statement

Online platforms contain a large number of movie reviews. Understanding the overall opinion of viewers manually can take considerable time.

The objective of this project is to develop a machine learning system that automatically classifies movie reviews as positive or negative based on their text.

## 3. Project Objectives

* Understand the basics of Natural Language Processing.
* Analyze and preprocess movie review text.
* Clean text by removing unnecessary characters and formatting.
* Convert textual data into numerical features for machine learning.
* Train a machine learning model to classify sentiment.
* Evaluate the model's performance using suitable metrics.
* Build an interactive web application using Streamlit.
* Test the system using different movie reviews.

## 4. Technologies Used

| Technology                   | Purpose                                     |
| ---------------------------- | ------------------------------------------- |
| Python                       | Main programming language                   |
| Pandas                       | Data loading and manipulation               |
| NumPy                        | Numerical operations                        |
| Scikit-learn                 | Feature extraction and machine learning     |
| NLP and regular expressions  | Text cleaning and preprocessing             |
| TF-IDF                       | Converting text into numerical features     |
| Support Vector Machine (SVM) | Sentiment classification                    |
| Matplotlib and Seaborn       | Data visualization, if used in the notebook |
| Jupyter Notebook             | Development and experimentation             |
| Streamlit                    | Web application                             |
| Git and GitHub               | Version control and project hosting         |

## 5. Dataset

The project uses an IMDb movie review dataset containing review text and sentiment labels.

The dataset contains two sentiment categories:

* **Positive:** Reviews expressing a favorable opinion about a movie.
* **Negative:** Reviews expressing an unfavorable opinion about a movie.

The dataset used during development contained 50,000 rows and four columns:

* `review`: Original movie review.
* `sentiment`: Positive or negative label.
* `review_length`: Length of the review.
* `cleaned_review`: Processed review text used for analysis.

The dataset was checked for missing values before model development.

## 6. What Is Sentiment Analysis?

Sentiment analysis is a Natural Language Processing technique that identifies the opinion or emotion expressed in text.

For example:

**Positive review:**
"The movie was excellent, and the acting was amazing."

Predicted sentiment: Positive

**Negative review:**
"The story was boring, and the movie was disappointing."

Predicted sentiment: Negative

The goal is to help computers recognize whether written feedback expresses a positive or negative opinion.

## 7. Project Workflow

The project follows these main steps:

1. **Data collection:** Load the IMDb movie review dataset.
2. **Data exploration:** Examine the dataset, columns, labels, and review lengths.
3. **Data cleaning:** Remove unnecessary text elements and normalize the reviews.
4. **Feature extraction:** Convert the cleaned text into numerical features using TF-IDF.
5. **Model training:** Train an SVM classifier on the prepared data.
6. **Model evaluation:** Evaluate the model using suitable classification metrics.
7. **Model saving:** Save the trained model for future predictions.
8. **Application development:** Build a Streamlit interface for users.
9. **Testing:** Enter sample movie reviews and check the predictions.

## 8. Text Preprocessing

Text preprocessing prepares raw reviews for machine learning.

The project includes a cleaned review column called `cleaned_review`. The preprocessing process uses text-cleaning operations to remove unnecessary elements and make the reviews more suitable for feature extraction.

Depending on the preprocessing code, these operations can include:

* Converting text to lowercase.
* Removing HTML tags.
* Removing unnecessary punctuation or symbols.
* Normalizing extra whitespace.
* Preparing the cleaned text for TF-IDF vectorization.

Cleaning helps reduce irrelevant variation in the text and makes the input more consistent.

## 9. Feature Extraction Using TF-IDF

Machine learning models cannot directly understand raw text in the same way humans do. Text must first be converted into numerical features.

**TF-IDF (Term Frequency–Inverse Document Frequency)** assigns numerical weights to words based on their frequency in a review and their distribution across the collection of reviews.

For example, words such as "excellent", "boring", "amazing", and "disappointing" may provide useful information for classifying sentiment.

TF-IDF converts the cleaned reviews into numerical vectors that the SVM classifier can process.

## 10. Machine Learning Model: SVM

The project uses a **Support Vector Machine (SVM)** classifier.

SVM is a supervised machine learning algorithm that learns to separate different classes using a decision boundary. In this project, the classes are positive and negative movie reviews.

After training on labeled reviews, the model can predict the sentiment of a new review based on the features extracted from its text.

The trained model is saved as:

`models/svm_sentiment_model.pkl`

This saved file allows the application to load the trained model instead of training it every time the application starts.

## 11. Model Evaluation

A sentiment classification model should be evaluated using appropriate metrics.

The following metrics can be used in this project:

* **Accuracy:** The proportion of all predictions that are correct.
* **Precision:** The proportion of predicted positive or negative cases that are correct for the selected class.
* **Recall:** The proportion of actual cases of a class that the model correctly identifies.
* **F1-score:** A combined measure of precision and recall.
* **Confusion matrix:** A table showing correct and incorrect predictions for each class.

Actual metric values should be added after running the final evaluation code. The results depend on the train-test split, preprocessing steps, and model settings.

## 12. Streamlit Web Application

The project includes a Streamlit web application located at:

`app/app.py`

The application allows users to enter a movie review and obtain a sentiment prediction.

### How the application works

1. The user enters a movie review in the text input area.
2. The application loads the saved model and any required feature-extraction objects.
3. The review is processed using the same preprocessing and vectorization approach used during training.
4. The model predicts the sentiment.
5. The predicted class is displayed to the user.

The interface makes it easier to test the model without manually running notebook cells.

## 13. Project Folder Structure

```text
Sentiment_Analysis/
│
├── app/
│   └── app.py
│
├── models/
│   └── svm_sentiment_model.pkl
│
├── results/
│
├── IMDb_Sentiment_Analysis.ipynb
├── .gitignore
└── README.md
```

The exact files in the `results` folder may vary depending on which evaluation outputs and visualizations have been saved.

The original dataset is excluded from this example structure because large datasets are generally better kept outside a standard GitHub code repository.

## 14. How to Run the Project

### Prerequisites

Install Python and Git. The saved model and all required preprocessing or vectorization objects must be available locally.

### Step 1: Clone the repository

```bash
git clone https://github.com/Shivam3831143/IMDb-Sentiment-Analysis.git
cd IMDb-Sentiment-Analysis
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn
```

Install any additional libraries used by the notebook or application.

### Step 4: Run the application

From the project root directory, execute:

```powershell
python -m streamlit run app/app.py
```

Open the local URL displayed in the terminal. Enter a movie review and check the predicted sentiment.

**Note:** The model file and any required TF-IDF vectorizer or preprocessing objects must be present at the paths expected by the application. If these files are not included in GitHub, they must be obtained separately.

## 15. Limitations

* The model predicts only positive or negative sentiment.
* It may misunderstand sarcasm, irony, or complicated expressions.
* A review containing both praise and criticism may be difficult to classify correctly.
* Performance depends on the quality and representativeness of the training dataset.
* Predictions may be less reliable for text that differs significantly from IMDb movie reviews.
* The model does not explain every factor that influenced its prediction.

## 16. Future Improvements

The project can be extended in several ways:

* Compare SVM with Logistic Regression, Naive Bayes, and other classifiers.
* Evaluate different TF-IDF settings and preprocessing approaches.
* Add confusion matrix and classification report visualizations.
* Improve the interface with clearer prediction displays.
* Explore transformer-based models such as BERT.
* Test the model on reviews from different movie platforms.
* Add automated tests and a requirements file.
* Deploy the application online for public access.

## 17. Conclusion

The IMDb Sentiment Analysis project demonstrates how Natural Language Processing and machine learning can be used to classify movie reviews automatically.

The project covers the complete workflow, from loading and cleaning text data to extracting TF-IDF features, training an SVM classifier, saving the model, and building a Streamlit application.

It provides practical experience with text preprocessing, feature extraction, supervised learning, model evaluation, and application development. Further experiments and evaluation can help improve the model's performance and reliability.

## 18. References

1. IMDb Datasets: https://developer.imdb.com/non-commercial-datasets/
2. Scikit-learn Documentation: https://scikit-learn.org/stable/
3. Pandas Documentation: https://pandas.pydata.org/docs/
4. Streamlit Documentation: https://docs.streamlit.io/
5. Python Documentation: https://docs.python.org/3/

---

**Project:** IMDb Sentiment Analysis Using Machine Learning
**GitHub Repository:** https://github.com/Shivam3831143/IMDb-Sentiment-Analysis
