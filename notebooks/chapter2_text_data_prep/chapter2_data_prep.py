import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tokenizer
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regex experimentation
    """)
    return


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
    return (tokenized_vocab,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simple Tokenizer
    """)
    return


@app.cell
def _(re):
    class SimpleTokenizerV1:
        def __init__(self, vocab):
            # Vocab is a dict of the fomat {word:idx}
            self.str_to_int = vocab
            self.int_to_str = {i:s for s,i in vocab.items()}

        def encode(self, text):
            pattern = re.compile(r'([,.:;?_!"()\']|--|\s)')
            preprocessed = re.split(pattern, text)
            striped = [item.strip() for item in preprocessed if item.strip()]
            ids = [self.str_to_int[s] for s in striped]
            return ids

        def decode(self, ids):
            text = " ".join([self.int_to_str[i] for i in ids])

            cleaned_text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
            return cleaned_text

    return (SimpleTokenizerV1,)


@app.cell
def _(SimpleTokenizerV1, tokenized_vocab):
    tokentizer_v1 = SimpleTokenizerV1(tokenized_vocab)
    tokenized_test_text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
    print(tokentizer_v1.encode(tokenized_test_text))
    return tokenized_test_text, tokentizer_v1


@app.cell
def _(tokenized_test_text, tokentizer_v1):
    _ids = tokentizer_v1.encode(tokenized_test_text)
    print(tokentizer_v1.decode(_ids))
    return


@app.cell(disabled=True)
def _(tokentizer_v1):
    print(tokentizer_v1.encode("Hello, do you like tea?"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Handling unknown words and end of sentence
    """)
    return


@app.cell
def _(preprocessed):
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab_v2 = {token:integer for integer,token in enumerate(all_tokens)}
    print(len(vocab_v2.items()))
    return (vocab_v2,)


@app.cell
def _(vocab_v2):
    for i, item in enumerate(list(vocab_v2.items())[-5:]):
        print(item)
    return


@app.cell
def _(re):
    class SimpleTokenizerV2:
        def __init__(self, vocab):
            # Vocab is a dict of the fomat {word:idx}
            self.str_to_int = vocab
            self.int_to_str = {i:s for s,i in vocab.items()}

        def encode(self, text):
            pattern = re.compile(r'([,.:;?_!"()\']|--|\s)')
            preprocessed = re.split(pattern, text)
            striped = [item.strip() for item in preprocessed if item.strip()]
            unknowns = [item if item in self.str_to_int else "<|unk|>" for item in striped]
            ids = [self.str_to_int[s] for s in unknowns]
            return ids

        def decode(self, ids):
            text = " ".join([self.int_to_str[i] for i in ids])

            cleaned_text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
            return cleaned_text

    return (SimpleTokenizerV2,)


@app.cell
def _():
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    sample_text_v2 = " <|endoftext|> ".join((text1, text2))
    print(sample_text_v2)
    return (sample_text_v2,)


@app.cell
def _(SimpleTokenizerV2, sample_text_v2, vocab_v2):
    tokentizer_v2 = SimpleTokenizerV2(vocab_v2)
    print(tokentizer_v2.encode(sample_text_v2))
    return (tokentizer_v2,)


@app.cell
def _(sample_text_v2, tokentizer_v2):
    print(tokentizer_v2.decode(tokentizer_v2.encode(sample_text_v2)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Byte Pair Encoding
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Orignally proposed by _Philip Gage_ in **1994**
        - Focused on **compression**
        - Replaces the highest frequency _pair of bytes_ with a single byte not in original data
        - Use a lookup table to recreate the original dataset
        - Modified to not just replace _pair of bytes_ but any contigous sequence of characters
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example

    #### Original
    - Text: `aaabdaaabac`
        - Most frequent pair: `aa`; replace with `Z`
    - Step 1 text: `ZabdZabac` + `Z=aa`
        - Now most frequent pair is `ab`
    - Step 2 text: `ZYdZYac` + `Y=ab` & `Z=aa`
        - Now it `ac` once so we can stop now
    - Another option is _recursive byte-pair_ encoding where we replace `ZY` with `X`
    ```
    XdXac
    X=ZY
    Y=ab
    Z=aa
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Modified
    - For language modeling by encoding **plaintext** --> **tokens**
    - Compression is not the priority
    - Process:
        - Treat set of unique characters as 1-character-long n-grams (initial token)
        - Most frequent _pair of adjacent_ tokens is merged into new, longer n-grams and all instances of the pair with new token
        - This is repeated until vocab size is restricted to a prescribed size
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        graph TD
        Start(["Input String:<br/>aaabdaaabac"]) --> Init[Initialize Vocabulary]

        subgraph S1 [Step 1: Base Characters]
        direction LR
            Vocab1["Vocab (Size 4):<br/>a:0, b:1, d:2, c:3"]
            Enc1["Encoding:<br/>0,0,0,1,2,0,0,0,1,0,3"]
        end

        Init --> S1
        S1 --Set aa=4--> S2

        subgraph S2 [Step 2: Vocab Size 5]
        direction LR
            Vocab2["Vocab (Size 5):<br/>a:0, b:1, d:2, c:3, aa:4"]
            Enc2["Encoding:<br/>4,0,1,2,4,0,1,0,3"]
        end

        S2 -- Set ab=5 --> S3

        subgraph S3 [Step 3: Vocab Size 6]
        direction LR
            Vocab3["Vocab (Size 6):<br/>... aa:4, ab:5"]
            Enc3["Encoding:<br/>4,5,2,4,5,0,3"]
        end

        S3 -- "Set aaab=6, aaabd=7" --> S4

        subgraph S4 [Step 4: Vocab Size 8]
        direction LR
            Vocab4["Vocab (Size 8):<br/>... aaab:6, aaabd:7"]
            Enc4["Encoding:<br/>7,6,0,3"]
        end

        style S2 fill:#f9f,stroke:#333
        style S3 fill:#bbf,stroke:#333
        style S4 fill:#bfb,stroke:#000
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Byte-level BPE

    Convert any text to UTF-8 and then treat it as a stream of bytes
    """)
    return


@app.cell
def _():
    import tiktoken
    print(f"{tiktoken.__version__=}")
    return (tiktoken,)


@app.cell
def _(tiktoken):
    bpe_tokenizer = tiktoken.get_encoding("gpt2")
    return (bpe_tokenizer,)


@app.cell
def _(bpe_tokenizer):
    _text = """Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."""
    _integers = bpe_tokenizer.encode(_text, allowed_special={"<|endoftext|>"})
    print(_integers)
    print(bpe_tokenizer.decode(_integers))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
