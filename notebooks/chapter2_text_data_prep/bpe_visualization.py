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
    print(os.getcwd()) 
    return (os,)


@app.cell
def _(os):
    # add current directory to path since I run marimo from the base directory
    import sys
    module_dir = os.path.join(os.getcwd(), 'notebooks', 'chapter2_text_data_prep')
    sys.path.insert(0, module_dir)
    # print(sys.path)
    return (module_dir,)


@app.cell
def _():
    from tiktoken_educational import SimpleBytePairEncoding, bpe_train_steps, visualise_tokens

    return SimpleBytePairEncoding, bpe_train_steps, visualise_tokens


@app.cell
def _(SimpleBytePairEncoding, bpe_train_steps, module_dir, visualise_tokens):
    def train_simple_encoding():
        gpt2_pattern = (
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        with open(f"{module_dir}/tiktoken_educational.py", 'r') as f:
            data = f.read()

        mergeable_ranks, training_frames, training_captions = bpe_train_steps(
            data,
            vocab_size=600,
            pat_str=gpt2_pattern,
        )
        enc = SimpleBytePairEncoding(pat_str=gpt2_pattern, mergeable_ranks=mergeable_ranks)

        tokens = enc.encode("hello world", visualise=None)
        encoding_steps = enc.encode_steps("hello world")
        assert enc.decode(tokens) == "hello world"
        assert enc.decode_bytes(tokens) == b"hello world"
        assert enc.decode_tokens_bytes(tokens) == [b"hello", b" world"]

        return {
            "enc": enc,
            "tokens": tokens,
            "training_animation": visualise_tokens(
                training_frames,
                captions=training_captions,
                title="BPE training progression (first 50 words)",
            ),
            "encoding_animations": [
                {
                    "word": step["word"],
                    "html": visualise_tokens(
                        step["steps"],
                        captions=step["captions"],
                        title=f"Encoding {step['word']!r}",
                    ),
                }
                for step in encoding_steps
            ],
        }

    return (train_simple_encoding,)


@app.cell
def _(mo, train_simple_encoding):
    result = train_simple_encoding()
    return mo.vstack(
        [
            mo.md("# BPE visualisation"),
            mo.md(f"Encoded tokens for `hello world`: `{result['tokens']}`"),
            mo.md("## Training animation"),
            mo.Html(result["training_animation"]),
            mo.md("## Encoding animation"),
            *[
                mo.vstack(
                    [
                        mo.md(f"### {animation['word']!r}"),
                        mo.Html(animation["html"]),
                    ]
                )
                for animation in result["encoding_animations"]
            ],
        ]
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
