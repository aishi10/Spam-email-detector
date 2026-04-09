# Spam Demo

This folder contains a small end-to-end spam classification demo:

- dataset
- a text classifier trained on that dataset
- a prediction CLI
- a Flask website for browser-based testing
- a benchmark script that evaluates a real held-out SMS spam split from the parent project

## Files

- `emails.csv`
- `train_spam_classifier.py` trains the synthetic demo model and saves `spam_model.joblib`
- `predict_email.py` predicts a single message from the saved model
- `app.py` runs the website
- `benchmark_real_dataset.py` evaluates a real SMS spam test split using the parent dataset

## Setup

From the project root:

```bash
cd "/Users/chaku/Desktop/Spam Filtering/toy_spam_demo"
```

This writes `emails.csv` in the same folder.

## Train the toy model

```bash
python3 train_spam_classifier.py
```

This saves:

- `spam_model.joblib`
- `confusion_matrix.png`

## Predict a single message

```bash
python3 predict_email.py "Congratulations! You won a lottery!"
```

## Run the website

```bash
PORT=5006 python3 app.py
```

Then open:

- `http://127.0.0.1:5006`

## Benchmark on a real SMS dataset

```bash
python3 benchmark_real_dataset.py
```

This uses the parent folder's real SMS spam data and reports a true held-out test-split accuracy.

