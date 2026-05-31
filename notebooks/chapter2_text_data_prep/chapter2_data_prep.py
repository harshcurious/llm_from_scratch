import marimo

__generated_with = "0.23.8"
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
    preprocessed = [
        item.strip() for item in re.split(pattern, raw_text) if item.strip()
    ]
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
    tokenized_vocab = {word: i for i, word in enumerate(vocab)}
    print({word: i for word, i in tokenized_vocab.items() if i < 50})
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
            self.int_to_str = {i: s for s, i in vocab.items()}

        def encode(self, text):
            pattern = re.compile(r'([,.:;?_!"()\']|--|\s)')
            preprocessed = re.split(pattern, text)
            striped = [item.strip() for item in preprocessed if item.strip()]
            ids = [self.str_to_int[s] for s in striped]
            return ids

        def decode(self, ids):
            text = " ".join([self.int_to_str[i] for i in ids])

            cleaned_text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
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
    vocab_v2 = {token: integer for integer, token in enumerate(all_tokens)}
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
            self.int_to_str = {i: s for s, i in vocab.items()}

        def encode(self, text):
            pattern = re.compile(r'([,.:;?_!"()\']|--|\s)')
            preprocessed = re.split(pattern, text)
            striped = [item.strip() for item in preprocessed if item.strip()]
            unknowns = [
                item if item in self.str_to_int else "<|unk|>" for item in striped
            ]
            ids = [self.str_to_int[s] for s in unknowns]
            return ids

        def decode(self, ids):
            text = " ".join([self.int_to_str[i] for i in ids])

            cleaned_text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
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
def _(bpe_tokenizer):
    _text = """Akwirw ier"""
    _integers = bpe_tokenizer.encode(_text, allowed_special={"<|endoftext|>"})
    print(_integers)
    print(bpe_tokenizer.decode(_integers))
    print([bpe_tokenizer.decode([i]) for i in _integers])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sliding Window data sampling
    """)
    return


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches


    def create_llm_prediction_diagram(
        text_phrase, output_filename="llm_prediction.png"
    ):
        # Split the phrase into individual words
        words = text_phrase.split()
        num_rows = len(words)

        # Setup figure and axis with a dark charcoal background
        fig, ax = plt.subplots(figsize=(12, 7), facecolor="#11110b")
        ax.set_facecolor("#11110b")

        # Set coordinate limits (0 to 100 on both axes for predictable plotting)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.axis("off")

        # Constants for layout
        start_y = 90
        row_spacing = 9
        start_x = 15
        word_spacing = 5.5  # Approximate spacing horizontal shift per word

        # Global Text Styles
        text_color = "#888883"
        highlight_box_color = "#aaaaaa"

        # Draw Static Labels
        ax.text(
            1,
            start_y,
            "Text\nsample:",
            color="#ffffff",
            fontsize=12,
            fontweight="bold",
            va="top",
        )
        ax.text(
            1,
            start_y - (row_spacing * 4.5),
            "Input the\nLLM\nreceives",
            color="#ffffff",
            fontsize=12,
            fontweight="bold",
            va="center",
        )

        # Draw Explanatory Text Blocks
        ax.text(
            105,
            start_y - (row_spacing * 3),
            "The LLM can't\naccess words past\nthe target.",
            color="#ffffff",
            fontsize=13,
            fontweight="bold",
            va="center",
            ha="left",
        )
        ax.text(
            105,
            start_y - (row_spacing * 7),
            "Target to\npredict",
            color="#ffffff",
            fontsize=13,
            fontweight="bold",
            va="center",
            ha="left",
        )

        # Iterate through each step of the prediction sequence
        for row_idx in range(num_rows):
            current_y = start_y - (row_idx * row_spacing)

            # 1. Print the base sequence words for this row
            x_offset = start_x
            word_positions = []

            for w in words:
                txt = ax.text(
                    x_offset,
                    current_y,
                    w,
                    color=text_color,
                    fontsize=12,
                    va="center",
                    ha="left",
                )
                word_positions.append((x_offset, len(w)))
                x_offset += (
                    len(w) * 1.3 + word_spacing
                )  # Estimate width based on character count

            # 2. Draw Context Bounding Box (Input the LLM receives)
            # The context grows by one word each row
            context_end_idx = row_idx
            if context_end_idx >= 0:
                box_start_x = word_positions[0][0] - 1
                # Calculate width to encompass all words up to the context index
                last_word_pos = word_positions[context_end_idx]
                box_end_x = last_word_pos[0] + (last_word_pos[1] * 1.3) + 1
                box_width = box_end_x - box_start_x

                rect_context = patches.FancyBboxPatch(
                    (box_start_x, current_y - 2.5),
                    box_width,
                    5,
                    boxstyle="round,pad=0.3",
                    linewidth=1.5,
                    edgecolor=highlight_box_color,
                    facecolor="none",
                )
                ax.add_patch(rect_context)

            # 3. Draw Target Bounding Box (Target to predict)
            # The target is always the word immediately following the context
            target_idx = row_idx + 1
            if target_idx < num_rows:
                target_word_pos = word_positions[target_idx]
                t_start_x = target_word_pos[0] - 1
                t_width = (target_word_pos[1] * 1.3) + 2

                rect_target = patches.FancyBboxPatch(
                    (t_start_x, current_y - 2.5),
                    t_width,
                    5,
                    boxstyle="round,pad=0.3",
                    linewidth=1,
                    edgecolor="#666661",
                    facecolor="none",
                )
                ax.add_patch(rect_target)

        # 4. Programmatic Arrow Annotations
        # Arrow pointing to "Input the LLM receives"
        ax.annotate(
            "",
            xy=(14, start_y - (row_spacing * 4.2)),
            xytext=(10, start_y - (row_spacing * 4.5)),
            arrowprops=dict(
                arrowstyle="->", color="#888883", connectionstyle="arc3,rad=-0.2"
            ),
        )

        # Arrow for "The LLM can't access words past the target"
        ax.annotate(
            "",
            xy=(66, start_y - (row_spacing * 4)),
            xytext=(105, start_y - (row_spacing * 3)),
            arrowprops=dict(
                arrowstyle="->", color="#888883", connectionstyle="arc3,rad=0.3"
            ),
        )

        # Arrow for "Target to predict"
        ax.annotate(
            "",
            xy=(70, start_y - (row_spacing * 4.2)),
            xytext=(105, start_y - (row_spacing * 7)),
            arrowprops=dict(
                arrowstyle="->", color="#888883", connectionstyle="arc3,rad=-0.3"
            ),
        )

        # Save and optimize layout
        plt.tight_layout()
        # plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.show()


    # Execute generation
    _phrase = "LLMs learn to predict one word at a time"
    create_llm_prediction_diagram(_phrase)
    return


@app.cell
def _(bpe_tokenizer, file_path):
    with open(file_path, "r", encoding="utf-8") as _f:
        _raw_text = _f.read()
    enc_text = bpe_tokenizer.encode(_raw_text)
    print(len(enc_text))
    return (enc_text,)


@app.cell
def _(enc_text):
    enc_sample = enc_text[50:]
    return (enc_sample,)


@app.cell
def _(enc_sample):
    context_size = 4  # 1
    x = enc_sample[:context_size]
    y = enc_sample[1 : context_size + 1]
    print(f"x: {x}")
    print(f"y: {y}")
    return (context_size,)


@app.cell
def _(context_size, enc_sample):
    for _i in range(1, context_size + 1):
        _context = enc_sample[:_i]
        _desired = enc_sample[_i]
        print(_context, "---->", _desired)
    return


@app.cell
def _(bpe_tokenizer, context_size, enc_sample):
    for _i in range(1, context_size + 1):
        _context = enc_sample[:_i]
        _desired = enc_sample[_i]
        print(
            bpe_tokenizer.decode(_context),
            "---->",
            bpe_tokenizer.decode([_desired]),
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![Sliding Window Visualization](public/sliding_window_torch.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Custom Dataset Class

    <https://gemini.google.com/share/21eddadaf9f8>
    """)
    return


@app.cell
def _():
    import torch
    from torch.utils.data import Dataset, DataLoader


    class GPTDatasetV1(Dataset):
        def __init__(self, txt, tokenizer, max_length, stride):
            self.input_ids = []
            self.target_ids = []

            token_ids = tokenizer.encode(txt)

            for _i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[_i : _i + max_length]
                target_chunk = token_ids[_i + 1 : _i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]

    return DataLoader, GPTDatasetV1, torch


@app.cell
def _(DataLoader, GPTDatasetV1, tiktoken):
    def create_dataloader_v1(
        txt,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    ):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(
            txt, tokenizer, max_length=max_length, stride=stride
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        return dataloader

    return (create_dataloader_v1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dataloader
    """)
    return


@app.cell
def _(create_dataloader_v1, file_path):
    with open(file_path, "r", encoding="utf-8") as _f:
        raw_text_full = _f.read()

    dataloader = create_dataloader_v1(
        raw_text_full, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    return data_iter, raw_text_full


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > input_size is generally 256
    """)
    return


@app.cell
def _(data_iter):
    second_batch = next(data_iter)
    print(second_batch)
    return


@app.cell
def _(create_dataloader_v1, raw_text_full):
    _dataloader = create_dataloader_v1(
        raw_text_full, batch_size=1, max_length=4, stride=2, shuffle=False
    )
    _data_iter = iter(_dataloader)
    _first_batch = next(_data_iter)
    print(_first_batch)
    _second_batch = next(_data_iter)
    print(_second_batch)
    return


@app.cell
def _(create_dataloader_v1, raw_text_full):
    _dataloader = create_dataloader_v1(
        raw_text_full, batch_size=1, max_length=2, stride=2, shuffle=False
    )
    _data_iter = iter(_dataloader)
    _first_batch = next(_data_iter)
    print(_first_batch)
    _second_batch = next(_data_iter)
    print(_second_batch)
    return


@app.cell
def _(create_dataloader_v1, raw_text_full):
    _dataloader = create_dataloader_v1(
        raw_text_full, batch_size=1, max_length=8, stride=2, shuffle=False
    )
    _data_iter = iter(_dataloader)
    _first_batch = next(_data_iter)
    print(_first_batch)
    _second_batch = next(_data_iter)
    print(_second_batch)
    return


@app.cell
def _(create_dataloader_v1, raw_text_full):
    _dataloader = create_dataloader_v1(
        raw_text_full, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    _data_iter = iter(_dataloader)
    _first_batch = next(_data_iter)
    print(_first_batch[0].shape, "\n", _first_batch, "\n")
    _second_batch = next(_data_iter)
    print(_second_batch)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating Token Embeddings

    ![alt](public/encoding.png)
    """)
    return


@app.cell
def _(torch):
    input_ids = torch.tensor([2, 3, 5, 1])

    _vocab_size = 6
    _output_dim = 3

    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(_vocab_size, _output_dim)
    print(f"{embedding_layer.weight.shape = }")
    print(embedding_layer.weight)
    print(f"{embedding_layer(torch.tensor([3])).shape = }")
    print(embedding_layer(torch.tensor([3])))
    return embedding_layer, input_ids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - The weight matrix of the embedding layer contains small,
    random values.

    - These values are optimized during LLM
    training as part of the LLM optimization itself.

    - Embedding layer <===> OHE + Matrix mul in fully connected layer
        - More details at <https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/03_bonus_embedding-vs-matmul/embeddings-and-linear-layers.ipynb>
    """)
    return


@app.cell
def _(embedding_layer, input_ids):
    print(f"{input_ids = }")

    print(f"{embedding_layer(input_ids).shape = }")

    print(embedding_layer(input_ids))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Encoding word positions

    - Self-attention has no concept of token ordering
    - So we inject additional position info to the LLM


    **Position aware embeddings**
    1. Relative positional embeddings
        -  Focus on distance b/w tokens
        -  Generalizes well to various (new to model) lengths
    2. Absolute positional embeddings
        - We _add unique embedding_ to each input embedding to convey it's position in the sequence
        - ![alt](public/abs_pos_embedding.png)

    > Choice depends on specific application and data being processed
    >
    > GPT 2-3 used absolute positional embeddings but they were optimized during training (ie not fixed)
    """)
    return


@app.cell
def _(torch):
    final_vocab_size = 50257
    final_output_dim = 256
    token_embedding_layer = torch.nn.Embedding(final_vocab_size, final_output_dim)
    return final_output_dim, token_embedding_layer


@app.cell
def _(create_dataloader_v1, raw_text):
    max_length = 4
    final_dataloader = create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )
    final_data_iter = iter(final_dataloader)
    inputs, targets = next(final_data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    return inputs, max_length


@app.cell
def _(inputs, token_embedding_layer):
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)
    return (token_embeddings,)


@app.cell
def _(final_output_dim, max_length, torch):
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, final_output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)
    return (pos_embeddings,)


@app.cell
def _(pos_embeddings, token_embeddings):
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
