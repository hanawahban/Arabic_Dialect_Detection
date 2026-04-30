# Arabic Dialect Identification using TF-IDF and Machine Learning

## Overview
This project focuses on classifying Arabic text into different dialects using machine learning techniques. It explores both word-level and character-level TF-IDF representations to capture linguistic and morphological patterns in dialectal Arabic.

## Problem
Arabic dialect identification is a challenging NLP task due to significant variation in vocabulary, spelling, and structure across regions. This project aims to build a model that can accurately classify text into dialect categories such as Gulf, Egyptian, Levantine, Moroccan, and Tunisian.

## Approach
- Preprocessed and normalized Arabic text
- Extracted features using:
  - Word-level TF-IDF
  - Character-level TF-IDF
- Trained 5 machine learning models (Naive Bayes, Logistic Regression, KNN, Random Forest, Linear SVC)
- Compared performance between feature representations and machine learning models

## Results
- Achieved approximately 82% classification accuracy
- Word-level Naive Bayes was most accurate in capturing dialectal differences

## Tech Stack
- Python
- Scikit-learn
- Pandas
- NumPy
