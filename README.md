# Fake News Detection with Machine Learning

Group course project (CSE445 Machine Learning, North South University) that classifies news articles as fake or real using TF-IDF text features and three classifiers.

## Dataset
- `Fake.csv` (fake articles) and `True.csv` (real articles), with title, text and subject columns
- About 44,900 articles in total; an 80/20 train–test split (random_state = 1) gives 8,980 test articles (4,682 fake, 4,298 real)
- The CSV files are stored with Git LFS

## Method
Each model is a scikit-learn pipeline trained on the article text:
CountVectorizer → TfidfTransformer → classifier
1. Logistic Regression
2. Linear Support Vector Machine (LinearSVC)
3. Multinomial Naive Bayes

Models are evaluated with accuracy, confusion matrices, and per-class precision, recall and F1-score.

## Results (test set: 8,980 articles)
| Model                   | Accuracy | Macro F1-score | Misclassified |
|-------------------------|----------|----------------|---------------|
| Linear SVM              | 99.39%   | 0.99           | 55            |
| Logistic Regression     | 98.73%   | 0.99           | 114           |
| Multinomial Naive Bayes | 94.08%   | 0.94           | 532           |

## Repository contents
- `Fake News Detection.ipynb`: full code and results
- `Fake.csv`, `True.csv`: dataset (Git LFS)
- `Project Proposal Presentation-….pptx`, `Fake News Detection Presentation-….pptx`: proposal and final presentations
- `1.png` – `7a.png`: image files from the project

## How to run
1. Install Git LFS, clone the repository and run `git lfs pull`
2. `pip install pandas numpy scikit-learn`
3. Open the notebook in Jupyter and run all cells

## Tools
Python, pandas, NumPy, scikit-learn
