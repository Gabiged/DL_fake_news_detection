import re
import spacy
import string
import contractions
import nltk
import pandas as pd
import numpy as np
from collections import Counter
from matplotlib.pylab import plt
from sklearn.metrics import confusion_matrix, classification_report
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm

import seaborn as sns
import spacy.cli

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def plot_histogram(data_df, column_name, label_fake="Fake", label_real="Real"):
    """
    Plot a histogram for a specific column in the DataFrame, distinguishing
    between fake and real news.
    """
    plt.figure(figsize=(10, 6))

    plt.hist(
        data_df[data_df["type"] == 0][column_name],
        bins=20,
        color="red",
        alpha=0.7,
        label=label_fake,
    )
    plt.hist(
        data_df[data_df["type"] == 1][column_name],
        bins=20,
        color="blue",
        alpha=0.7,
        label=label_real,
    )

    plt.title(f"Histogram of {column_name} for Fake and Real News")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def text_remove_url(text):
    """Removing URL's"""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def text_remove_html_tags(text):
    clean = re.sub(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});", "", text)
    return clean


def text_remove_non_ascii(text):
    """Use a regular expression to remove non-ASCII characters"""
    ascii_only = re.sub(r"[^\x00-\x7f]", "", text)
    return ascii_only


def text_lower_case(text):
    return text.lower()


def text_expand_contractions(text: str):
    """
    Expand contractions in a given text, except for specified exceptions.
    """
    exceptions = {"U.S.", "U.S.A."}
    words = text.split()

    expanded_words = []
    for word in words:
        if word.upper() in exceptions:
            expanded_words.append(word)
        else:
            expanded_words.append(contractions.fix(word))
    expanded_text = " ".join(expanded_words)

    return expanded_text


def text_remove_stopwords(text):
    """Remove stopwords"""
    stop_words = set(stopwords.words("english"))
    words = text.split()
    clean_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(clean_words)


def text_remove_special_character(text):
    """
    Remove special characters, leave multiple dots and exclamation marks.
    Remove extra white spaces.
    Remove single dots at the end of the sentence
    """
    cleaned_text = re.sub(r"[^a-zA-Z0-9.!]+", "", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    cleaned_text = re.sub(r"(?<=\w)\.", "", cleaned_text)
    cleaned_text = re.sub("\[.*?\]", "", cleaned_text)
    cleaned_text = re.sub("\\W", " ", cleaned_text)
    cleaned_text = re.sub("\n", "", cleaned_text)
    cleaned_text = re.sub("\w*\d\w*", "", cleaned_text)
    return cleaned_text


def data_cleaning(text):
    text = text_remove_url(text)
    text = text_remove_html_tags(text)
    text = text_remove_non_ascii(text)
    text = text_lower_case(text)
    text = text_expand_contractions(text)
    text = text_remove_stopwords(text)
    text = text_remove_special_character(text)
    return text


def get_top_ngrams(text_column: str, n: int, top=10):
    """
    Get the top n-grams from text column.
    """
    all_ngrams = [
        ngram
        for text in text_column
        for ngram in ngrams(word_tokenize(text), n)
    ]
    ngram_counts = Counter(all_ngrams)
    return ngram_counts.most_common(top)


def generate_wordcloud(text: str, title: str):
    """
    Generate and display a word cloud for the given text.
    """
    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


def filter_df_by_date_pattern(df):
    """
    Filter DataFrame rows based on a specific date pattern.
    The goal is to find annomalies.
    """
    pattern = r"(([A-Z][a-z]+) \d{1,2}, \d{4})|(\d{1,2}-([A-Z][a-z]+)-\d{2})"
    date_mismatch = []

    for index, row in df.iterrows():
        if not re.search(pattern, row["date"], re.IGNORECASE):
            date_mismatch.append(index)

    df.drop(date_mismatch, inplace=True)
    df.drop(columns=["date"], axis=1, inplace=True)
    return df


def drop_rows_with_empty_or_whitespace_cells(df):
    """
    Drop rows from a DataFrame with empty or whitespace cells.
    """
    empty_or_whitespace_cells = df.applymap(
        lambda x: x.strip() == "" if isinstance(x, str) else False
    ).any(axis=1)
    df.drop(df[empty_or_whitespace_cells].index, inplace=True)
    return df


def drop_duplicate_rows_by_column(df):
    """
    Drop rows from a DataFrame based on duplicates in a specific text column.
    """
    df.drop_duplicates(subset=["text"], keep=False, inplace=True)
    return df


def calculate_text_length(df):
    df["text_length"] = df["text"].apply(
        lambda text: len(text) - text.count(" ")
    )
    return df


def count_punctuation(df):
    df["punctuation_count"] = df["text"].apply(
        lambda text: sum(1 for char in text if char in string.punctuation)
    )
    return df


def count_words(df):
    df["word_count"] = df["text"].apply(lambda text: len(text.split()))
    return df


def replace_names_with_person(df):
    nlp = spacy.load("en_core_web_lg")

    def replace_names(text):
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON" or ent.text.lower() in [
                "trump",
                "donald",
            ]:
                text = text.replace(ent.text, "PERSON")
        return text

    df["text"] = df["text"].apply(replace_names)
    return df


def exclude_before_reuters(df):
    """Remove everything before (Reuters) - if present and Reuters itself"""
    pattern_before = re.compile(r".*?\(Reuters\) - (.*)", re.DOTALL)
    pattern_after = re.compile(r"(.*)(Reuters)(.*)", re.IGNORECASE)

    def extract_reuters(text):
        text = str(text)
        match = pattern_before.search(text)
        text = match.group(1) if match else text
        text = pattern_after.sub(r"\1\3", text)
        return text.strip()

    df["text"] = df["text"].apply(extract_reuters)

    return df


def concat_title(df):
    df["text"] = df[["title", "text"]].apply(" ".join, axis=1)
    df.drop(columns=["title"], axis=1, inplace=True)
    return df


def remove_special_characters(df):
    """
    Remove special characters.
    Remove extra white spaces.
    Remove digits.
    """
    df["text"] = df["text"].apply(lambda text: re.sub(r"\[.*?\]", "", text))

    df["text"] = df["text"].apply(
        lambda text: re.sub(r"[^a-zA-Z0-9 ]", "", text)
    )
    df["text"] = df["text"].apply(
        lambda text: re.sub(r"\s+", " ", text).strip()
    )
    df["text"] = df["text"].apply(lambda text: re.sub(r"\W", " ", text))
    df["text"] = df["text"].apply(
        lambda text: re.sub(r"\d|\b\w*\d\w*\b", "", text)
    )
    return df


def remove_url(df):
    """Removing URL's"""
    df["text"] = df["text"].apply(
        lambda text: re.sub(r"https?://\S+|www\.\S+", "", text)
    )
    return df


def remove_html_tags(df):
    df["text"] = df["text"].apply(
        lambda text: re.sub(
            r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});", "", text
        )
    )
    return df


def remove_non_ascii(df):
    """Use a regular expression to remove non-ASCII characters"""
    df["text"] = df["text"].apply(
        lambda text: re.sub(r"[^\x00-\x7f]", "", text)
    )
    return df


def remove_weekdays(df):
    weekdays_pattern = (
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    )
    df["text"] = df["text"].apply(
        lambda text: re.sub(weekdays_pattern, "", text, flags=re.IGNORECASE)
    )

    return df


def expand_contractions(df):
    """
    Expand contractions in a given text, except for specified exceptions.
    """
    exceptions = {"U.S.", "U.S.A."}
    df["text"] = df["text"].apply(
        lambda text: " ".join(
            contractions.fix(word) if word.lower() not in exceptions else word
            for word in text.split()  # type: ignore
        )
    )
    return df


def nltk_tokenization(df):
    df["text"] = df["text"].apply(lambda text: word_tokenize(text))
    return df


def remove_stopwords(df):
    """Remove stopwords"""
    stop_words = set(stopwords.words("english"))
    df["text"] = df["text"].apply(
        lambda text: [
            word
            for word in word_tokenize(text)
            if word.lower() not in stop_words
        ]
    )
    return df


def stem_text(df):
    stemmer = PorterStemmer()
    df["text"] = df["text"].apply(
        lambda tokens: " ".join(stemmer.stem(word) for word in tokens)
    )
    return df


def plot_confusion_matrix(
    y_true, y_pred, labels=None, title="Confusion Matrix", cmap="Blues"
):
    """
    Plot a confusion matrix.
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    labels (array-like, optional): List of label names.
    title (str, optional): Title of the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = np.unique(y_true)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted", fontsize=10)
    plt.ylabel("True", fontsize=10)
    plt.title(title, fontsize=12)
    plt.show()


def plot_feature_importance(
    coef_text,
    feature_names_text,
    coef_numeric=None,
    feature_names_numeric=None,
    title="Feature Importance",
):
    """
    Plot feature importance for text and (optionally) numeric features.
    coef_text (array-like): Coefficients for feature importance for text features.
    coef_numeric (array-like, optional): Coefficients for feature importance for numeric features.
    feature_names_text (array-like): Names of the text features.
    feature_names_numeric (array-like, optional): Names of the numeric features.
    """
    if coef_numeric is not None and feature_names_numeric is None:
        raise ValueError(
            "feature_names_numeric cannot be None if coef_numeric is provided."
        )

    top_text_features = 10
    top_numeric_features = 5

    sorted_text_idx = np.argsort(coef_text)
    top_text_features_idx = sorted_text_idx[-top_text_features:]
    top_text_features = feature_names_text[top_text_features_idx]
    top_text_coef = coef_text[top_text_features_idx]

    plt.figure(figsize=(10, 6))

    plt.barh(
        range(len(top_text_coef)),
        top_text_coef,
        align="center",
        color="blue",
        label="Text Features",
    )

    if coef_numeric is not None:
        sorted_numeric_idx = np.argsort(coef_numeric)
        top_numeric_features_idx = sorted_numeric_idx[-top_numeric_features:]
        top_numeric_features = feature_names_numeric[top_numeric_features_idx]  # type: ignore
        top_numeric_coef = coef_numeric[top_numeric_features_idx]

        plt.barh(
            np.arange(
                len(top_text_coef), len(top_text_coef) + len(top_numeric_coef)
            ),
            top_numeric_coef,
            align="center",
            color="orange",
            label="Numeric Features",
        )
        plt.yticks(
            np.arange(len(top_text_features) + len(top_numeric_features)),
            np.concatenate((top_text_features, top_numeric_features)),  # type: ignore
        )
        for i, v in enumerate(top_text_coef):
            plt.text(v, i, f"{v:.2f}", va="center")

        for i, v in enumerate(top_numeric_coef):
            plt.text(v, i + len(top_text_coef), f"{v:.2f}", va="center")

    else:
        plt.yticks(np.arange(len(top_text_features)), top_text_features)

    for i, coef in enumerate(top_text_coef):
        plt.text(coef, i, f"{coef:.2f}", ha="left", va="center")

    plt.xlabel("Importance", fontsize=10)
    plt.ylabel("Feature", fontsize=9)
    plt.title(title, fontsize=12)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()


def explain_text_features_lime(
    input_pipeline,
    X_test,
    y_test,
    sample_size=1000,
    target_labels=["Fake_News", "Real_News"],
):
    sample_indices_test = np.random.choice(
        len(X_test), sample_size, replace=False
    )
    X_test_sample = X_test["text"].iloc[sample_indices_test]
    y_test_sample = y_test.iloc[sample_indices_test]

    explainer = LimeTextExplainer(class_names=target_labels)

    aggregate_feature_importance = {}
    for text in tqdm(
        X_test_sample, desc="Explaining Instances", unit="instance"
    ):
        exp_model = explainer.explain_instance(
            text, input_pipeline.predict_proba, num_features=10
        )
        for feature, weight in exp_model.as_list():
            aggregate_feature_importance[feature] = (
                aggregate_feature_importance.get(feature, 0) + abs(weight)
            )

    sorted_features = sorted(
        aggregate_feature_importance.items(), key=lambda x: x[1], reverse=True
    )

    top_features = [feature for feature, _ in sorted_features[:10]]
    top_importance = [importance for _, importance in sorted_features[:10]]

    plt.figure(figsize=(12, 6))
    bars = plt.barh(top_features, top_importance, color="skyblue")
    for bar, coef in zip(bars, top_importance):
        plt.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{coef:.2f}",
            va="center",
        )
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Top Features and Their Importance Scores")
    plt.gca().invert_yaxis()
    plt.show()


def single_lime_explain(
    input_pipeline,
    X_test,
    y_test,
    index,
    target_labels=["Fake_News", "Real_News"],
):
    explainer = LimeTextExplainer(class_names=target_labels)
    exp_rf = explainer.explain_instance(
        X_test["text"].iloc[index],
        input_pipeline.predict_proba,
        num_features=10,
    )

    print("Document id: %d" % index)
    print(
        "Probability(fake news) =",
        input_pipeline.predict_proba([X_test["text"].iloc[index]])[0, 0],
    )
    print("True class: %s" % target_labels[y_test.iloc[index].item()])

    fig = exp_rf.as_pyplot_figure()
    exp_rf.show_in_notebook(text=False)


def class_report_conf_matrix(
    pipeline_input, X_test, y_test, target_labels=["Fake_news", "Real_news"]
):
    """
    Evaluate a classification model and print the classification report
    and plot the confusion matrix.
    """
    y_pred = pipeline_input.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=target_labels,
        title="Confusion Matrix",
    )


def plot_text_feature_importance(pipeline):
    """
    Plot feature importance for text-only components of a pipeline.
    """
    if "classifier" in pipeline.named_steps:
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "coef_"):
            coef = classifier.coef_[0]
            if "vectorizer" in pipeline.named_steps:
                vectorizer = pipeline.named_steps["vectorizer"]
                if hasattr(vectorizer, "get_feature_names_out"):
                    feature_names = vectorizer.get_feature_names_out()
                    plot_feature_importance(
                        coef,
                        feature_names,
                        title="Text Feature Importance",
                    )


# Deep learning preprocessing


def remove_special_characters_dl(df):
    """
    Remove special characters, leave multiple dots and exclamation marks.
    Remove extra white spaces.
    Remove single dots at the end of the sentence
    """
    df["text"] = df["text"].apply(
        lambda text: re.sub(r"[^a-zA-Z0-9.!]+", " ", text)
    )
    df["text"] = df["text"].apply(
        lambda text: re.sub(r"\s+", " ", text).strip()
    )
    df["text"] = df["text"].apply(lambda text: re.sub(r"(?<=\w)\.", "", text))
    return df


def process_lower_case_dl(df):
    """Change characters to lower case, keep all caps words"""

    def process_text(text):
        processed_words = []
        words = re.split(r"(\s+|\.+)", text)
        for word in words:
            if word.isupper():
                processed_words.append(word)
            else:
                processed_words.append(word.lower())
        return " ".join(processed_words)

    df["text"] = df["text"].apply(process_text)
    return df


def data_preprocess(df):
    df = filter_df_by_date_pattern(df)
    df = drop_rows_with_empty_or_whitespace_cells(df)
    df = drop_duplicate_rows_by_column(df)
    df = exclude_before_reuters(df)
    df = concat_title(df)
    df = remove_html_tags(df)
    df = remove_non_ascii(df)
    df = remove_special_characters_dl(df)
    df = process_lower_case_dl(df)
    return df
