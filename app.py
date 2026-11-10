"""
UIC CS 411 - Project 03: Naive Bayes Text Classifier from Scratch
Multinomial Naive Bayes with Laplace smoothing (alpha = 1), log-space scoring.
Pure Python standard library only.
"""

import os
import csv
import math
import re
from collections import Counter

# ---------------------------------------------------------------- model state
MODEL = {
    "vocab": set(),                 # V: all unique training words
    "word_counts": {                # count(w, c)
        "positive": Counter(),
        "negative": Counter(),
    },
    "total_words": {"positive": 0, "negative": 0},   # N_c
    "doc_counts": {"positive": 0, "negative": 0},    # sentences per class
    "priors": {"positive": 0.0, "negative": 0.0},    # P(c)
}

CLASSES = ("positive", "negative")


def tokenize(text):
    """Lowercase and split into word tokens (letters/digits, keeps apostrophes)."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def load_csv(path):
    """Read a text,label CSV into a list of (text, label) tuples."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if text and label in CLASSES:
                rows.append((text, label))
    return rows


def train_model(train_file="data/train.csv"):
    """
    Train the Naive Bayes classifier on the training dataset.
    Computes vocabulary, per-class word counts, N_c, and class priors.
    Likelihoods with Laplace smoothing are computed from these counts
    on demand in word_likelihood().
    """
    rows = load_csv(train_file)

    MODEL["vocab"] = set()
    MODEL["word_counts"] = {c: Counter() for c in CLASSES}
    MODEL["total_words"] = {c: 0 for c in CLASSES}
    MODEL["doc_counts"] = {c: 0 for c in CLASSES}

    for text, label in rows:
        MODEL["doc_counts"][label] += 1
        for w in tokenize(text):
            MODEL["word_counts"][label][w] += 1      # every occurrence counts
            MODEL["vocab"].add(w)

    for c in CLASSES:
        MODEL["total_words"][c] = sum(MODEL["word_counts"][c].values())

    n_docs = sum(MODEL["doc_counts"].values())
    for c in CLASSES:
        MODEL["priors"][c] = MODEL["doc_counts"][c] / n_docs if n_docs else 0.0

    return MODEL


def word_likelihood(word, c):
    """P(w | c) = (count(w, c) + 1) / (N_c + |V|)"""
    num = MODEL["word_counts"][c][word] + 1
    den = MODEL["total_words"][c] + len(MODEL["vocab"])
    return num / den


def score_sentence(text, verbose=False):
    """
    Return {class: log score}. Words not in V are ignored.
    score(c) = log P(c) + sum(log P(w | c)) over in-vocabulary tokens.
    """
    tokens = tokenize(text)
    known = [w for w in tokens if w in MODEL["vocab"]]
    unknown = [w for w in tokens if w not in MODEL["vocab"]]

    scores = {}
    for c in CLASSES:
        p = MODEL["priors"][c]
        scores[c] = math.log(p) if p > 0 else float("-inf")
        for w in known:
            scores[c] += math.log(word_likelihood(w, c))

    if verbose:
        print(f"  Tokens:        {tokens}")
        print(f"  Ignored (OOV): {unknown}")
        for c in CLASSES:
            p = MODEL["priors"][c]
            print(f"  log P({c}) = {math.log(p) if p > 0 else float('-inf'):.4f}")
        for w in known:
            row = "  ".join(f"P({w}|{c[:3]})={word_likelihood(w, c):.5f}" for c in CLASSES)
            print(f"    {row}")
        for c in CLASSES:
            print(f"  score({c}) = {scores[c]:.4f}")
    return scores


def predict(text, verbose=False):
    """
    Predict sentiment class for an input sentence.
    Returns: 'positive' or 'negative'
    Tie-breaking: higher prior P(c); if priors equal, 'positive'.
    """
    scores = score_sentence(text, verbose=verbose)
    pos, neg = scores["positive"], scores["negative"]
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    if MODEL["priors"]["negative"] > MODEL["priors"]["positive"]:
        return "negative"
    return "positive"


def evaluate_model(test_file="data/test.csv"):
    """
    Evaluate on a test set. Positive is the 'positive' class.
    Returns {"accuracy": float, "confusion_matrix": {"TP","TN","FP","FN"}}
    """
    rows = load_csv(test_file)
    cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for text, label in rows:
        pred = predict(text)
        if label == "positive":
            cm["TP" if pred == "positive" else "FN"] += 1
        else:
            cm["TN" if pred == "negative" else "FP"] += 1
    total = sum(cm.values())
    accuracy = (cm["TP"] + cm["TN"]) / total if total else 0.0
    return {"accuracy": accuracy, "confusion_matrix": cm}


def print_model_summary():
    print("=" * 60)
    print("MODEL SUMMARY (from training data)")
    print("=" * 60)
    print(f"Vocabulary size |V|: {len(MODEL['vocab'])}")
    for c in CLASSES:
        print(f"{c:>8}: sentences={MODEL['doc_counts'][c]:<4} "
              f"N_c={MODEL['total_words'][c]:<6} P(c)={MODEL['priors'][c]:.4f}")


def show_word(word):
    """Show exact counts and smoothed likelihoods for one word."""
    word = word.lower()
    V = len(MODEL["vocab"])
    print(f"Word: '{word}'  (|V| = {V})")
    for c in CLASSES:
        cnt = MODEL["word_counts"][c][word]
        n_c = MODEL["total_words"][c]
        print(f"  {c:>8}: count={cnt}  P(w|c) = ({cnt}+1)/({n_c}+{V}) "
              f"= {(cnt + 1) / (n_c + V):.6f}")


def main():
    train_file = "data/train.csv"
    test_file = "data/test.csv"
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Missing data/train.csv or data/test.csv")
        return

    train_model(train_file)
    print_model_summary()

    result = evaluate_model(test_file)
    cm = result["confusion_matrix"]
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Confusion matrix: TP={cm['TP']} TN={cm['TN']} FP={cm['FP']} FN={cm['FN']}")

    print("\nInteractive mode. Type a sentence to classify (step-by-step),")
    print("'word <w>' to inspect a word's counts, or 'quit' to exit.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        if line.lower().startswith("word "):
            show_word(line[5:].strip())
            continue
        print(f"Prediction: {predict(line, verbose=True)}")


if __name__ == "__main__":
    main()
