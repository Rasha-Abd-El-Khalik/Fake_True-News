# 📰 Fake vs True News Classifier (English NLP)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 📌 Overview
This project classifies **English news articles** into **Fake** or **True** using Natural Language Processing (NLP) and Machine Learning.  
It combines text preprocessing, Word2Vec embeddings, and multiple classifiers to achieve high accuracy.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| Preprocessing | Cleaning, tokenization, stopword removal, normalization |
| Embeddings | Word2Vec trained with Gensim |
| Models | XGBoost, Logistic Regression, Random Forest |
| Evaluation | Accuracy, Precision, Recall, F1-score |
| Visualization | Matplotlib + Seaborn plots |
| Export | Save trained models for later predictions |

---

## 📊 Results

- **XGBoost**
  - Validation Accuracy: **98.0%**
  - Test Accuracy: **97.8%**

- **Logistic Regression**
  - Train Accuracy: **96.9%**
  - Test Accuracy: **96.3%**

- **Random Forest**
  - Training Accuracy: **100%**
  - (Test results not re-uploaded due to notebook cell removal)

⚡ **XGBoost gave the best overall performance** with strong generalization and balanced metrics.

---

Run the notebook or load the trained model in Python to classify new articles.

## ⚠️ Important Note

The file word2vec_model.model is not included in this repository due to GitHub file size limits.

👉 Options:

Train the model yourself (see train_word2vec.ipynb)
Or
[Download pre-trained Word2Vec model](https://drive.google.com/file/d/1pZrxH1gXuTYoNm08FdwpNibh98HqMxdG/view?usp=sharing)
