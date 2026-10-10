"""AIL 7024 Assignment 3 -- Authorship Attribution with Naive Bayes (STARTER CODE).

Fill in every function marked TODO. Do not change the CONFIGURATION block: the
train/test split is already made for you, and those constants fix the rest.

Add any extra classes, methods or helpers your design needs, but do not rename
anything already here or change its signature, since those are what will be
checked. If your design does not use one of them, leave it unimplemented.

Run with:  python authorship_attribution.py
"""
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import os
import string
import time
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# ============================== CONFIGURATION ==============================

ALPHA = 1.0
SMALL_N_PER_AUTHOR = 100
BALANCED_PER_AUTHOR = 4226
CURVE_SIZES: tuple[int, ...] = (SMALL_N_PER_AUTHOR, 300, 1000, 3000, BALANCED_PER_AUTHOR)
SUBSAMPLE_SEED = 0
GENERATION_LENGTH = 20
GENERATION_SEED = 0
AUTHORS: tuple[str, ...] = ("EAP", "HPL", "MWS")
OUTPUT_DIR = os.path.join("outputs", "nb_authorship")
TRAIN_CSV = "data/nb_authorship/train.csv"    # both provided; download using the link in the assignment
TEST_CSV = "data/nb_authorship/test.csv"

# ================================== DATA ===================================


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read TRAIN_CSV and TEST_CSV. The split is already made; do not re-split."""
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    return (train_df,test_df)


def sample_per_author(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw exactly n sentences for each author, without replacement."""
    classes = df['author'].unique()
    n_classes = len(classes)
    #?? make this size per author
    # i'm hardcoding for three classes right now
    train_set = pd.DataFrame()
    for c in classes:
        train_set = pd.concat([train_set,df[df['author']==c].sample(n=n,random_state=seed)])
    return train_set.sample(frac=1,random_state=seed,ignore_index=True)


# ================================= METRICS =================================


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Fraction of predictions that match the true author."""
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str]) -> np.ndarray:
    """A 3x3 array of counts, true author by predicted author, ordered by AUTHORS.

    You may use sklearn.metrics.confusion_matrix here: it already uses this layout.
    """
    cm_per_trnsize = [] # confusion matrix
    #?? hardcoding for 3 classes for now
    classes = np.unique(y_true)
    pred0 = (y_pred == classes[0]);true0 = (y_true == classes[0])
    pred1 = (y_pred == classes[1]);true1 = (y_true == classes[1])
    pred2 = (y_pred == classes[2]);true2 = (y_true == classes[2])
    tp0 = np.sum(pred0&true0);err01 = np.sum(true0&pred1);err02 = np.sum(true0&pred2)
    err10 = np.sum(true1&pred0);tp1 = np.sum(pred1&true1);err12 = np.sum(true1&pred2)
    err20 = np.sum(true2&pred0);err21 = np.sum(true2&pred1);tp2 = np.sum(pred2&true2)
    cm_per_trnsize.append(np.array([[tp0,err01,err02],[err10,tp1,err12],[err20,err21,tp2]]))
    return np.array(cm_per_trnsize)


def precision_recall_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    """Precision, recall and F1 for each author, plus a macro-average row.

    Columns: author, precision, recall, f1 -- one row per author in AUTHORS
    order, then a final row with author = "macro".
    """
    raise NotImplementedError("TODO")


# ================================ TOKENISERS ===============================


# Provided: this section is already implemented for you.


def tokenize_words(text: str) -> list[str]:
    """Lowercase, split on whitespace, strip surrounding punctuation, drop empties."""
    stripped = (word.strip(string.punctuation) for word in text.lower().split())
    return [word for word in stripped if word]


def tokenize_word_ngrams(text: str, n: int) -> list[str]:
    """Runs of n consecutive words joined by a space; n=1 matches tokenize_words."""
    words = tokenize_words(text)
    if n == 1:
        return words
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def make_preprocessor(remove_stopwords: bool, stem: bool) -> Callable[[str], str]:
    """Return a text -> text function applying stopword removal and/or stemming."""
    if not (remove_stopwords or stem):
        return lambda text: text
    import nltk
    nltk.download('stopwords',quiet=True)
    # ------
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    stops = set(stopwords.words("english")) if remove_stopwords else set()
    stemmer = PorterStemmer() if stem else None
    cache: dict[str, str] = {}
    # assert False, stops

    def stem_word(word: str) -> str:
        if word not in cache:
            cache[word] = stemmer.stem(word)
        return cache[word]

    def preprocess(text: str) -> str:
        words = [w for w in tokenize_words(text) if w not in stops]
        if stemmer is not None:
            words = [stem_word(w) for w in words]
        return " ".join(words)

    return preprocess


def add_token_column(df: pd.DataFrame, tokenizer: Callable[[str], list[str]],
                     preprocessor: Callable[[str], str] | None = None) -> pd.DataFrame:
    """Attach the `tokens` column that fit() consumes."""
    texts = df["text"] if preprocessor is None else df["text"].map(preprocessor)
    out = df.copy()
    out["tokens"] = texts.map(tokenizer)
    return out


# ================================== MODEL ==================================


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = ALPHA) -> None:
        self.alpha = alpha

    def _preprocess(self,tokens_col):
        n_features = len(self.vocab)
        n_samples = len(tokens_col)
        token_embeddings = np.zeros((n_samples,n_features))
        ## !! add counts for unigram token embeddings
        for i, row in enumerate(tokens_col):
            for token in row:
                #@ ignore oov words for now
                if token in self.vocab:
                    token_embeddings[i][self.vocab[token]] += 1
        return token_embeddings

    def fit(self, df: pd.DataFrame, token_column: str = "tokens",
            label_column: str = "author") -> "MultinomialNaiveBayes":
        """Estimate the class priors and the smoothed token probabilities; return self."""
        tokens_col = df[token_column] # expecting this to be a list of string tokens
        n_samples = len(tokens_col)
        # construct ngram embedding from the list of tokens
        self.vocab = {token: idx for idx, token in enumerate(tokens_col.explode().unique())}
        n_features = len(self.vocab)
        labels = df[label_column]
        self.classes = labels.unique()
        n_classes = len(self.classes)
        ##
        token_embeddings = self._preprocess(tokens_col)
        ##
        self.priors = [(df[label_column]==c).sum()/n_samples for c in self.classes]
        self.theta = np.empty((n_classes,n_features))
        for k in range(n_classes):
            # hairy workaround: using pandas index for a numpy array
            # k_idx = df[labels==self.classes[k]].index
            k_idx = (labels == self.classes[k]).to_numpy()
            ##! will optimize this code later
            class_counts = np.sum(token_embeddings[k_idx],axis=0)
            self.theta[k] = (class_counts+self.alpha) / (np.sum(class_counts)+self.alpha*n_features)
        # print(f'learned prior probabilities: {self.priors}')
        # print(f'class conditional likelihood: {self.theta}')

    def predict(self, df: pd.DataFrame, token_column: str = "tokens") -> np.ndarray:
        """One predicted author per row of df."""
        tokens_col = df[token_column]
        token_embeddings = self._preprocess(tokens_col)
        log_theta = np.log(self.theta)
        log_priors = np.log(self.priors)
        log_posterior = (token_embeddings @ log_theta.T) + log_priors
        pred_indices = np.argmax(log_posterior,axis=1)
        return self.classes[pred_indices]


def oov_rate(model: MultinomialNaiveBayes, df: pd.DataFrame,
             token_column: str = "tokens") -> float:
    """Fraction of the tokens in df that are absent from the model's vocabulary."""
    oov_count = 0
    test_tokenized = add_token_column(df,tokenizer=lambda text: tokenize_word_ngrams(text,n=1))
    for token_row in test_tokenized[token_column]:
        for token in token_row:
            if token not in model.vocab:
                oov_count += 1
    # assert False, f'{oov_count} {len(model.vocab)}'
    return oov_count / len(model.vocab)


def generate_document(model: MultinomialNaiveBayes, author: str,
                      length: int = GENERATION_LENGTH,
                      seed: int = GENERATION_SEED) -> list[str]:
    """Sample `length` tokens for `author` from the fitted class-conditional model."""
    rng = np.random.default_rng(seed=seed)
    gen_result = []
    k = np.where(model.classes == author)[0]
    # generate a 20-token document
    for _ in range(length):
        # assert False, model.theta[k]
        gen_token_idx = np.argmax(rng.multinomial(n=1,pvals=model.theta[k]))
        # assert False, np.argmax(gen_token_idx)
        gen_token = list(model.vocab.keys())[gen_token_idx]
        gen_result.append(gen_token)
    gen_result = ' '.join(gen_result)
    return gen_result

# for plotting 3x3 confusion matrix
def plot_confusion_matrix(cm:np.ndarray,classes):
    cm = cm[0]
    assert len(cm) == 3 and len(cm[0]) == 3
    assert len(classes) == 3
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,8))
    ax.matshow(cm,cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticklabels(classes,fontsize=11)
    ax.set_yticklabels(classes,fontsize=11)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Actual Target')
    ax.set_xlabel('Predicted Label')
    threshold = cm.max() / 2
    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(
                j,i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                weight="bold",
            )
    plt.tight_layout()
    return fig

# =============================== EXPERIMENTS ===============================


def run_sample_size(train, test):
    # os.makedirs('outputs/nb_authorship',exist_ok=True)
    """Produce everything Section 3.1 asks for. Return the curve, so 3.3 can plot on it."""
    # train_tokenized = add_token_column(train,tokenizer=lambda text: tokenize_word_ngrams(text,n=1))
    test_tokenized = add_token_column(test,tokenizer=lambda text: tokenize_word_ngrams(text,n=1))
    nb_model = MultinomialNaiveBayes()
    train_sizes_per_author = [100,300,1000,3000,4226]
    test_accs, oov_rates = [], []
    cms = []
    print('### Naive Bayes ###')
    for trn_size in train_sizes_per_author:
        train_set = sample_per_author(train,n=trn_size,seed=0)
        train_set_tokenized = add_token_column(train_set,tokenizer=lambda text: tokenize_word_ngrams(text,n=1))  
        nb_model.fit(train_set_tokenized,token_column='tokens',label_column='author')
        pred_labels = nb_model.predict(test_tokenized,token_column='tokens')
        ##
        oov_rates.append(oov_rate(nb_model,test,token_column='tokens'))
        ##
        test_labels = test['author'].to_numpy()
        cms.append(confusion_matrix(test_labels,pred_labels))
        ##
        acc_test = accuracy(test_labels,pred_labels)
        test_accs.append(acc_test)
        print(f'Test accuracy = {acc_test:.4f}')
    # figure 3.1a
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.plot(train_sizes_per_author,test_accs,label='Naive Bayes')
    ax.set_xticks(train_sizes_per_author)
    ax.set_xticklabels(train_sizes_per_author)
    ax.set_xlabel('Number of sentences in train dataset per author')
    ax.set_ylabel('Test accuracy')
    ax.grid(alpha=0.5)
    # fig.savefig(f'{OUTPUT_DIR}/learning_curve.png',bbox_inches='tight')
    # figure 3.1b
    fig1 = plt.figure(figsize=(10,8))
    ax1 = fig1.add_subplot(111)
    ax1.clear()
    ax1.plot(train_sizes_per_author,oov_rates)
    ax1.set_xticks(train_sizes_per_author)
    ax1.set_xticklabels(train_sizes_per_author)
    ax1.set_xlabel('Number of sentences in train dataset per author')
    ax1.set_ylabel('OOV rate')
    ax1.grid(alpha=0.5)
    fig1.savefig(f'{OUTPUT_DIR}/oov_curve.png',bbox_inches='tight')
    plt.close(fig1)
    # figure 3.1c -- for 100 sentence and all sentence models
    fig1 = plot_confusion_matrix(cms[0],classes=nb_model.classes)
    fig1.savefig(f'{OUTPUT_DIR}/confusion_n100.png',bbox_inches='tight')
    fig1 = plot_confusion_matrix(cms[-1],classes=nb_model.classes)
    fig1.savefig(f'{OUTPUT_DIR}/confusion_full.png',bbox_inches='tight')
    plt.close(fig1)
    return fig


def run_generation(train, test):
    ## why have i defined the models here again?
    nb_model100 = MultinomialNaiveBayes()
    nb_modelfull = MultinomialNaiveBayes()
    classes = train['author'].unique()
    train_set100, train_setfull = pd.DataFrame(), pd.DataFrame()
    for c in classes:
        train_set100 = pd.concat([train_set100,train[train['author']==c][:100]])
        train_setfull = pd.concat([train_setfull,train[train['author']==c][:4226]])
        train_set100_tokenized = add_token_column(train_set100,tokenizer=lambda text: tokenize_word_ngrams(text,n=1))
        train_setfull_tokenized = add_token_column(train_setfull,tokenizer=lambda text: tokenize_word_ngrams(text,n=1))
    nb_model100.fit(train_set100_tokenized,token_column='tokens',label_column='author')
    nb_modelfull.fit(train_setfull_tokenized,token_column='tokens',label_column='author')
    """Produce everything Section 3.2 asks for."""
    with open(f'{OUTPUT_DIR}/generated_samples.txt','w',encoding='utf-8') as file:
        file.write('###### Model trained on dataset with 100 sentences per author ######\n')
        for c in nb_model100.classes:
            file.write(f'Author: {c}\n')
            gen_result = generate_document(nb_model100,c,length=20,seed=0)
            file.write(f'{gen_result}\n')
        file.write('###### Model trained on the full dataset ######\n')
        for c in nb_modelfull.classes:
            file.write(f'Author: {c}\n')
            gen_result = generate_document(nb_modelfull,c,length=20,seed=0)
            file.write(f'{gen_result}\n')


def run_logistic_comparison(train, test, nb_curve: plt.Figure):
    assert nb_curve is not None
    from sklearn.linear_model import LogisticRegression
    """Produce everything Section 3.3 asks for, drawing its curve beside nb_curve."""
    test_tokenized = add_token_column(test,tokenizer=lambda text: tokenize_word_ngrams(text,n=1))
    train_sizes_per_author = [100,300,1000,3000,4226]
    test_accs = []
    print('### Logistic Regression ###')
    for trn_size in train_sizes_per_author:
        ##? Logistic regression uses multinomial by default now
        logistic_model = LogisticRegression(max_iter=100)
        train_set = sample_per_author(train,n=trn_size,seed=0)
        train_set_tokenized = add_token_column(train_set,tokenizer=lambda text: tokenize_word_ngrams(text,n=1))  
        ## vectorize based on unigram counts
        tokens_col = train_set_tokenized['tokens']
        vocab = {token: idx for idx, token in enumerate(tokens_col.explode().unique())}
        n_features = len(vocab);n_samples = len(tokens_col)
        X_train = np.zeros((n_samples,n_features))
        X_test = np.zeros((len(test_tokenized['tokens']),n_features))
        for i, row in enumerate(tokens_col):
            for token in row:
                if token in vocab:
                    X_train[i][vocab[token]] += 1
        #?? vectorize test dataset using the train vocab
        for i, row in enumerate(test_tokenized['tokens']):
            for token in row:
                if token in vocab:
                    X_test[i][vocab[token]] += 1
        ##
        y_train = train_set_tokenized['author'].to_numpy()
        # assert False, X_train[0]
        #!! let's do l2 normalization on the train data
        X_train = X_train / (np.linalg.norm(X_train,axis=1,keepdims=True) + 1e-12)
        X_test = X_test / (np.linalg.norm(X_test,axis=1,keepdims=True) + 1e-12)
        logistic_model.fit(X_train,y_train)
        y_test = test_tokenized['author'].to_numpy()
        pred_labels = logistic_model.predict(X_test)
        acc_test = np.mean(y_test==pred_labels)
        print(f'Test accuracy = {acc_test:.4f}')
        test_accs.append(acc_test)
    ax = nb_curve.get_axes()[0]
    ax.plot(train_sizes_per_author,test_accs,label='Logistic Regression')
    ax.legend()
    nb_curve.savefig(f'{OUTPUT_DIR}/learning_curve.png',bbox_inches='tight')
    plt.close(nb_curve)


def run_representations(train, test):
    """Produce everything Section 3.4 asks for."""
    from wordcloud import WordCloud
    classes = train['author'].unique()
    all_text_per_author = []

    # create wordclouds
    def create_wordclouds(df,processed=False):
        for c in classes:
            all_text_per_author.append(' '.join(df[df['author']==c]['text'].dropna().astype('str')))
        # create wordcloud from the combined text
        wordclouds = [
            WordCloud(width=800,height=400,background_color='black',colormap='viridis').generate(text)
            for text in all_text_per_author
        ]
        for k,c in enumerate(classes):
            plt.figure(figsize=(10,8))
            plt.imshow(wordclouds[k],interpolation='bilinear')
            plt.axis('off')
            plt.savefig(f'{OUTPUT_DIR}/wordcloud_{c}_{'processed' if processed else 'raw'}.png',bbox_inches='tight')
            plt.close()

    create_wordclouds(train)
    ##Q how does stopword removal and stemming help?
    train_preprocessed = train.copy();test_preprocessed = test.copy()
    tokenizer = lambda text: tokenize_word_ngrams(text,n=1)
    preprocessor = make_preprocessor(remove_stopwords=True,stem=True)
    ## preprocess before tokenizing: just makes it shorter for me
    train_preprocessed['text'] = train_preprocessed['text'].map(preprocessor)
    test_preprocessed['text'] = test_preprocessed['text'].map(preprocessor)
    # now tokenize
    train_tokenized = add_token_column(train_preprocessed,tokenizer)
    test_tokenized = add_token_column(test_preprocessed,tokenizer)
    create_wordclouds(train_preprocessed,processed=True)
    #? test accuracy on stopword-removed/stemmed model
    nb_model = MultinomialNaiveBayes()
    nb_model.fit(train_tokenized,token_column='tokens',label_column='author')
    y_train_pred = nb_model.predict(train_tokenized,token_column='tokens')
    y_test_pred = nb_model.predict(test_tokenized,token_column='tokens')
    y_true = test_preprocessed['author']
    acc_train = accuracy(train_tokenized['author'],y_train_pred)
    acc_test = accuracy(y_true,y_test_pred)
    print(f'unigram vocabulary size = {len(nb_model.vocab)}')
    print(f'Train accuracy = {acc_train:.4f}')
    print(f'Test accuracy = {acc_test:.4f}')
    # generate samples
    print('###### Stopword removed model generated samples ######')
    for c in nb_model.classes:
        print(f'Author: {c}')
        gen_result = generate_document(nb_model,c,length=20,seed=0)
        print(f'{gen_result}')
    ## let's do bigram and trigram as well below (stopword-removed/stemmed)
    bigram_tokenizer = lambda text: tokenize_word_ngrams(text,n=2)
    train_bigram = add_token_column(train_preprocessed,bigram_tokenizer)
    test_bigram = add_token_column(test_preprocessed,bigram_tokenizer)
    nb_model_bigram = MultinomialNaiveBayes()
    nb_model_bigram.fit(train_bigram,token_column='tokens',label_column='author')
    y_bigram_train_pred = nb_model_bigram.predict(train_bigram,token_column='tokens')
    y_bigram_test_pred = nb_model_bigram.predict(test_bigram,token_column='tokens')
    acc_train_bigram = accuracy(train_bigram['author'],y_bigram_train_pred)
    acc_test_bigram = accuracy(y_true,y_bigram_test_pred)
    print(f'bigram vocabulary size = {len(nb_model_bigram.vocab)}')
    print(f'Train accuracy (bigram) = {acc_train_bigram:.4f}')
    print(f'Test accuracy (bigram) = {acc_test_bigram:.4f}')
    ## ===
    trigram_tokenizer = lambda text: tokenize_word_ngrams(text,n=3)
    train_trigram = add_token_column(train_preprocessed,trigram_tokenizer)
    test_trigram = add_token_column(test_preprocessed,trigram_tokenizer)
    nb_model_trigram = MultinomialNaiveBayes()
    nb_model_trigram.fit(train_trigram,token_column='tokens',label_column='author')
    y_trigram_train_pred = nb_model_trigram.predict(train_trigram,token_column='tokens')
    y_trigram_test_pred = nb_model_trigram.predict(test_trigram,token_column='tokens')
    acc_train_trigram = accuracy(train_trigram['author'],y_trigram_train_pred)
    acc_test_trigram = accuracy(y_true,y_trigram_test_pred)
    print(f'trigram vocabulary size = {len(nb_model_trigram.vocab)}')
    print(f'Train accuracy (trigram) = {acc_train_trigram:.4f}')
    print(f'Test accuracy (trigram) = {acc_test_trigram:.4f}')


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train, test = load_split()
    print(f"train={len(train)} ({BALANCED_PER_AUTHOR}/author, balanced)  test={len(test)}")

    started = time.time()
    nb_curve = run_sample_size(train, test)
    run_generation(train, test)
    run_logistic_comparison(train, test, nb_curve)
    run_representations(train, test)
    print(f"done in {time.time() - started:.1f}s; outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
