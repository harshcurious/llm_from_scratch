import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import os
    import requests

    return os, requests


@app.cell
def _(os, requests):
    file_path = "data/the-verdict.txt"

    if not os.path.exists(file_path):
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)
    return (file_path,)


@app.cell
def _():
    return


@app.cell
def _(file_path):
    with open(file_path, "r", encoding="utf-8") as f2:
        raw_text = f2.read()

    print("Total number of character:", len(raw_text))
    print(raw_text[:99])
    return (raw_text,)


@app.cell
def _():
    import re
    import string

    text = "Hello world. This is a test!"

    pattern = re.compile(r'([,.:;?_!"()\']|--|\s)')
    result = [item.strip() for item in re.split(pattern, text) if item.strip()]
    print(result)
    return pattern, re


@app.cell
def _(pattern, raw_text, re):
    preprocessed = [item.strip() for item in re.split(pattern, raw_text) if item.strip()]
    return (preprocessed,)


@app.cell
def _(preprocessed):
    print(preprocessed[:20])
    return


@app.cell
def _(preprocessed):
    print(len(preprocessed))
    return


@app.cell
def _(preprocessed):
    vocab = sorted(set(preprocessed))
    vocab_size = len(vocab)
    print(vocab_size)
    return (vocab,)


@app.cell
def _(vocab):
    tokenized_vocab = {word:i for i,word in enumerate(vocab)}
    print({word:i for word,i in tokenized_vocab.items() if i < 50})
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
