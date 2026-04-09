# Spam Demo

This folder contains a self-contained spam classification demo using a synthetic dataset.

Files:
- `generate_dataset.py` creates `emails.csv`
- `train_spam_classifier.py` trains a Naive Bayes spam classifier
- `predict_email.py` predicts new messages from the saved model
- `benchmark_real_dataset.py` evaluates a real held-out test split from the parent dataset

Run it:
```bash
cd "/Users/chaku/Desktop/Spam Filtering/toy_spam_demo"
python3 generate_dataset.py
python3 train_spam_classifier.py
python3 predict_email.py "Congratulations! You won a lottery!"
python3 benchmark_real_dataset.py
```
