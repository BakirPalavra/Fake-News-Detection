Fake News Detection using NLP and Machine Learning

A Big Data Analytics project that detects whether a news headline is Real or Fake using Natural Language Processing (NLP), feature engineering, and supervised machine learning models.

---

Project Overview

The rapid spread of fake news across digital platforms has become a major challenge in modern information ecosystems. This project develops a scalable machine learning pipeline capable of automatically classifying news headlines as either real or fake.

The system combines:
- NLP preprocessing
- TF-IDF vectorisation
- Handcrafted linguistic features
- Multiple machine learning classifiers
- Real-time prediction through a Streamlit web application

---

Features

- Large-scale NLP dataset processing
- Text cleaning and normalization
- TF-IDF vectorization with n-grams
- Metadata feature engineering
- Automatic best-model selection
- Real-time fake news prediction
- Streamlit deployment-ready interface

---

Dataset

This project uses a merged dataset constructed from two public Kaggle datasets:

1. WELFake Dataset
2. FakeNewsNet Dataset

After cleaning and deduplication, the final dataset contained approximately 95,000 headlines.

Label convention:
0 = Fake
1 = Real

---

Machine Learning Models

The following supervised learning models were trained and evaluated:

- Logistic Regression
- Complement Naive Bayes
- XGBoost Classifier

The system automatically selects and saves the best-performing model.

---

Technologies Used

| Category             | Technologies |

| Programming Language | Python 3 |
| NLP                  | NLTK |
| ML Libraries         | Scikit-learn, XGBoost |
| Data Processing      | Pandas, NumPy |
| Sparse Matrices      | SciPy |
| Visualisation        | Matplotlib, Seaborn |
| Deployment           | Streamlit |
| Environment          | Google Colab |

---

Feature Engineering

In addition to TF-IDF textual features, the project extracts handcrafted metadata features including:

- Exclamation mark count
- Question mark count
- Capital letter ratio
- All-caps word count
- Headline length
- Word count
- Average word length
- Number detection
- Clickbait keyword detection

These features improve classification performance by capturing stylistic patterns commonly associated with fake news.

---

Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

Streamlit Deployment

The project includes a Streamlit web application that allows users to input a news headline and receive an instant prediction.

Example:
Input:
"BREAKING: Secret government files finally exposed!"
Prediction:
Fake     Confidence: 54%


---

Project Structure

fake-news-detector/
│
├── app.py
├── model.pkl
├── tfidf.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
├── FakeNewsDetection.ipynb
└── sample_100_rows.csv

---

Future Improvements

Potential future enhancements include:

- BERT fine-tuning
- Full article classification
- Multilingual fake news detection
- Explainable AI using SHAP
- Real-time news scraping
- API deployment

---

# References

- OpenAI. (2025). *ChatGPT* [Large language model]. https://chatgpt.com
- Anthropic. (2025). *Claude* [Large language model]. https://claude.ai

---

# Author
Developed as part of a Big Data Analytics project focused on NLP and machine learning applications for misinformation detection.
