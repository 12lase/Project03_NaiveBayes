"""
UIC CS 411 - Project 03: Naive Bayes Text Classifier from Scratch
Multinomial Naive Bayes with Laplace smoothing (alpha = 1), log-space scoring.
Pure Python standard library only.
"""

import os                                  # check that files exist
import csv                                 # read the csv files
import math                                # log function
import re                                  # regex for splitting words
from collections import Counter            # dict that counts things

# ---------------------------------------------------------------- model state
MODEL = {                                  # everything the model learns lives here
    "vocab": set(),                 # V: all unique training words
    "word_counts": {                # count(w, c)
        "positive": Counter(),             # word -> times seen in positive
        "negative": Counter(),             # word -> times seen in negative
    },
    "total_words": {"positive": 0, "negative": 0},   # N_c
    "doc_counts": {"positive": 0, "negative": 0},    # sentences per class
    "priors": {"positive": 0.0, "negative": 0.0},    # P(c)
}

CLASSES = ("positive", "negative")         # the two labels


def tokenize(text):                        # turn a sentence into a list of words
    """Lowercase and split into word tokens (letters/digits, keeps apostrophes)."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())   # lowercase, then grab each word


def load_csv(path):                        # read a csv file
    """Read a text,label CSV into a list of (text, label) tuples."""
    rows = []                              # will hold (sentence, label) pairs
    with open(path, newline="", encoding="utf-8") as f:   # open the file
        for row in csv.DictReader(f):      # each row is a dict: text, label
            text = (row.get("text") or "").strip()            # sentence, no extra spaces
            label = (row.get("label") or "").strip().lower()  # label in lowercase
            if text and label in CLASSES:  # skip empty or bad rows
                rows.append((text, label)) # keep the good ones
    return rows                            # give back all pairs


def train_model(train_file="data/train.csv"):   # learn counts from training data
    """
    Train the Naive Bayes classifier on the training dataset.
    Computes vocabulary, per-class word counts, N_c, and class priors.
    Likelihoods with Laplace smoothing are computed from these counts
    on demand in word_likelihood().
    """
    rows = load_csv(train_file)            # load training sentences

    MODEL["vocab"] = set()                 # reset vocabulary
    MODEL["word_counts"] = {c: Counter() for c in CLASSES}   # reset word counts
    MODEL["total_words"] = {c: 0 for c in CLASSES}           # reset N_c
    MODEL["doc_counts"] = {c: 0 for c in CLASSES}            # reset sentence counts

    for text, label in rows:               # go through each training sentence
        MODEL["doc_counts"][label] += 1    # count the sentence for its class
        for w in tokenize(text):           # go through each word
            MODEL["word_counts"][label][w] += 1      # every occurrence counts
            MODEL["vocab"].add(w)          # add word to vocabulary

    for c in CLASSES:                      # for each class
        MODEL["total_words"][c] = sum(MODEL["word_counts"][c].values())   # N_c = all words in class

    n_docs = sum(MODEL["doc_counts"].values())   # total training sentences
    for c in CLASSES:                      # for each class
        MODEL["priors"][c] = MODEL["doc_counts"][c] / n_docs if n_docs else 0.0   # P(c) = class sentences / all

    return MODEL                           # done training


def word_likelihood(word, c):              # P(word | class)
    """P(w | c) = (count(w, c) + 1) / (N_c + |V|)"""
    num = MODEL["word_counts"][c][word] + 1                  # count + 1 (Laplace smoothing)
    den = MODEL["total_words"][c] + len(MODEL["vocab"])      # N_c + |V|
    return num / den                       # the probability


def score_sentence(text, verbose=False):   # log score for each class
    """
    Return {class: log score}. Words not in V are ignored.
    score(c) = log P(c) + sum(log P(w | c)) over in-vocabulary tokens.
    """
    tokens = tokenize(text)                # split sentence into words
    known = [w for w in tokens if w in MODEL["vocab"]]       # words we trained on
    unknown = [w for w in tokens if w not in MODEL["vocab"]] # new words, ignored

    scores = {}                            # one score per class
    for c in CLASSES:                      # score positive, then negative
        p = MODEL["priors"][c]             # P(c)
        scores[c] = math.log(p) if p > 0 else float("-inf")  # start with log prior
        for w in known:                    # for each known word
            scores[c] += math.log(word_likelihood(w, c))     # add log likelihood

    if verbose:                            # print the steps if asked
        print(f"  Tokens:        {tokens}")      # all words
        print(f"  Ignored (OOV): {unknown}")     # words not in vocabulary
        for c in CLASSES:                  # show each log prior
            p = MODEL["priors"][c]         # P(c)
            print(f"  log P({c}) = {math.log(p) if p > 0 else float('-inf'):.4f}")
        for w in known:                    # show each word's likelihoods
            row = "  ".join(f"P({w}|{c[:3]})={word_likelihood(w, c):.5f}" for c in CLASSES)
            print(f"    {row}")             # print that row
        for c in CLASSES:                  # show final scores
            print(f"  score({c}) = {scores[c]:.4f}")
    return scores                          # both class scores


def predict(text, verbose=False):          # pick positive or negative
    """
    Predict sentiment class for an input sentence.
    Returns: 'positive' or 'negative'
    Tie-breaking: higher prior P(c); if priors equal, 'positive'.
    """
    scores = score_sentence(text, verbose=verbose)           # get both scores
    pos, neg = scores["positive"], scores["negative"]        # pull them out
    if pos > neg:                          # positive wins
        return "positive"
    if neg > pos:                          # negative wins
        return "negative"
    if MODEL["priors"]["negative"] > MODEL["priors"]["positive"]:   # tie: bigger prior wins
        return "negative"
    return "positive"                      # still tied: default positive


def evaluate_model(test_file="data/test.csv"):   # test accuracy
    """
    Evaluate on a test set. Positive is the 'positive' class.
    Returns {"accuracy": float, "confusion_matrix": {"TP","TN","FP","FN"}}
    """
    rows = load_csv(test_file)             # load test sentences
    cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}   # confusion matrix counts
    for text, label in rows:               # go through each test sentence
        pred = predict(text)               # model's guess
        if label == "positive":            # actually positive
            cm["TP" if pred == "positive" else "FN"] += 1    # right = TP, wrong = FN
        else:                              # actually negative
            cm["TN" if pred == "negative" else "FP"] += 1    # right = TN, wrong = FP
    total = sum(cm.values())               # number of test sentences
    accuracy = (cm["TP"] + cm["TN"]) / total if total else 0.0   # correct / total
    return {"accuracy": accuracy, "confusion_matrix": cm}        # send results back


def print_model_summary():                 # show what was learned
    print("=" * 60)                        # divider line
    print("MODEL SUMMARY (from training data)")
    print("=" * 60)                        # divider line
    print(f"Vocabulary size |V|: {len(MODEL['vocab'])}")         # unique words
    for c in CLASSES:                      # per-class stats
        print(f"{c:>8}: sentences={MODEL['doc_counts'][c]:<4} "  # sentences, N_c, prior
              f"N_c={MODEL['total_words'][c]:<6} P(c)={MODEL['priors'][c]:.4f}")


def show_word(word):                       # look up one word
    """Show exact counts and smoothed likelihoods for one word."""
    word = word.lower()                    # match the lowercase vocab
    V = len(MODEL["vocab"])                # |V|
    print(f"Word: '{word}'  (|V| = {V})")  # header
    for c in CLASSES:                      # each class
        cnt = MODEL["word_counts"][c][word]    # times word seen in class
        n_c = MODEL["total_words"][c]          # N_c
        print(f"  {c:>8}: count={cnt}  P(w|c) = ({cnt}+1)/({n_c}+{V}) "   # show the formula with numbers
              f"= {(cnt + 1) / (n_c + V):.6f}")


def main():                                # runs when you do python app.py
    train_file = "data/train.csv"          # training path
    test_file = "data/test.csv"            # test path
    if not os.path.exists(train_file) or not os.path.exists(test_file):   # files must exist
        print("Missing data/train.csv or data/test.csv")
        return                             # stop if missing

    train_model(train_file)                # learn from training data
    print_model_summary()                  # show |V|, N_c, priors

    result = evaluate_model(test_file)     # test on test data
    cm = result["confusion_matrix"]        # the TP/TN/FP/FN counts
    print("\n" + "=" * 60)                 # divider line
    print("TEST SET EVALUATION")
    print("=" * 60)                        # divider line
    print(f"Accuracy: {result['accuracy']:.4f}")                 # accuracy
    print(f"Confusion matrix: TP={cm['TP']} TN={cm['TN']} FP={cm['FP']} FN={cm['FN']}")   # matrix

    print("\nInteractive mode. Type a sentence to classify (step-by-step),")   # instructions
    print("'word <w>' to inspect a word's counts, or 'quit' to exit.")
    while True:                            # keep asking until quit
        try:
            line = input("\n> ").strip()   # read what the user types
        except (EOFError, KeyboardInterrupt):   # Ctrl+C or Ctrl+D
            print()
            break                          # leave the loop
        if not line:                       # empty input
            continue                       # ask again
        if line.lower() in ("quit", "exit", "q"):   # user wants out
            break
        if line.lower().startswith("word "):        # "word great" style command
            show_word(line[5:].strip())    # show that word's counts
            continue                       # ask again
        print(f"Prediction: {predict(line, verbose=True)}")   # classify and show the steps


if __name__ == "__main__":                 # only run main when run directly
    main()
