import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    return Path, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Intro

    This notebook implements 4 variants of attention mechanism

    1. **Simplified self-attention**: basic implementation for intro
    2. **Self-attention**: self attention with trainable weights
    3. **Causal attention**: allows model to consider only prev and curr imputs in seq (temporal ordering)
    4. **Multi-head attention**: extension of _self-attention + causal attention_ ; attends to info from different representation spaces
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Long Seq Modeling problem

    Pre-LLM architecture without attention issues
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Consider a text translation model. Translating text word-by-word doesn't respect grammatical structures.
    To avoid this we usually used a DNN with encoder+decoder architecture. (Can think of encoder as reading the original language, and decoder as writing in the target language)

    RNN was the popular architecture for language translations. Explored in another notebook. Key idea:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(r"""
    graph LR

    S1[Sequential Input] --|encoder|--> S2[Hidden state] --|decoder|--> S3[Output]

    style S2 fill:#444400,stroke:#000
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    But the hidden state cannot be accessed at a later step! So only the info in current hidden state can be used, losing all other context. So translating paragraphs/books becomes a challenge. Especially translating code from one language to another!!!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Start of attentions mechanisms
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo):
    mo.md(rf"""
    ### Bahdanau attention

    - For RNNs
    - in 2015
    - Details: <https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html>

    {mo.image(Path("notebooks/chapter3_attention/public/Bahdanau.png"), style={"background-color": "#8D83F2"})}


    > **Key Insight:**
    >
    > - Decoder can selectively access different parts of the input sequence at each decoding step
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simplified Self-Attention

    Self-Attention without trainable weights
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Self** in self-attention
    >
    > - Ability to compute attention weights as per position in an input sequence
    > - Captures relationship between parts of input (eg words in sentences, pixels in an image)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    Let $X = (x^1, x^2, ..., x^T)$ be the sentence "_Your journey starts with one step_". It will look as follows as an input sequence

    {mo.image("notebooks/chapter3_attention/public/self_attention_input.png", height=100, width=500)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""
    We can generate a **context vector** for $x^2$ by assigning weights $\alpha_{21}$, $\alpha_{22}$, $\alpha_{23}$, ..., $\alpha_{T}$ corresponding to all the input vectors and matrix mult. We will call this **context vector** $z^2$. 
    """),
            mo.image(
                "notebooks/chapter3_attention/public/context_vector.png",
                height=300,
                width=450,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Context Vector**
    - enriched embedding of an input state based on other input vectors
    - (like a given word enriched by the other elements of the sentence)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    #### Code for $x^2$
    {mo.image("notebooks/chapter3_attention/public/fixed_weight_context_vector.png", height=250, width=500)}
    """)
    return


@app.cell
def _():
    import torch

    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ]
    )
    return inputs, torch


@app.cell
def _(inputs, torch):
    query = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0])
    print(f"{query = }, {attn_scores_2 = }")
    for _i, _x_i in enumerate(inputs):
        attn_scores_2[_i] = torch.dot(_x_i, query)
    print(f"\n{attn_scores_2 = }")
    return attn_scores_2, query


@app.cell
def _(inputs, query):
    inputs @ query
    return


@app.cell
def _(attn_scores_2):
    # Standard Normalization
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We more commonly use softmax for normalization
    - This also makes them positve
    - Closer to probabilities/relative importance
    """)
    return


@app.cell
def _(attn_scores_2, torch):
    def softmax_naive(x):
        # will have numerical instability with large or small values (float issues)
        return torch.exp(x) / torch.exp(x).sum(dim=0)


    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print("Attention weights:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())
    return


@app.cell
def _(attn_scores_2, torch):
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())
    return (attn_weights_2,)


@app.cell
def _(attn_weights_2, inputs, query, torch):
    # query = inputs[1]

    context_vec_2 = torch.zeros(query.shape)
    for _i, _x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[_i] * _x_i
    print(context_vec_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        "notebooks/chapter3_attention/public/second_context_vector.png",
        height=250,
        width=550,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Implementing the overall code

    1. Compute attentions scores
    2. Compute attention weights
    3. Compute context vectors
    """)
    return


@app.cell
def _(inputs):
    # Step1 attention scores
    # ... currently unnormalized
    attn_scores = inputs @ inputs.T
    print(attn_scores)
    return (attn_scores,)


@app.cell
def _(attn_scores, torch):
    # step2: normalization
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(attn_weights)
    return (attn_weights,)


@app.cell
def _(attn_weights, inputs):
    # Step3: context vector
    all_context_vecs = attn_weights @ inputs
    print(all_context_vecs)
    return


@app.cell
def _(mo):
    # summary
    mo.mermaid(r"""
    graph LR

    S1[Input] --|encoder|--> S2[Encoded Input] -- |simple attention| --> S3["Context Vector=</br>softmax(input @ input.T) @ inputs"] --|decoder|--> S5[Output]

    style S2 fill:#444400,stroke:#000
    style S3 fill:#9944AA,stroke:#A80
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adding trainable weights

    True self-attention
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - We will implement the version used in early GPT models
    - called _scaled dot-product attention_
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""
    - 3 trainable weight matrices: $W_q$, $W_k$, and $W_v$
    - Projects embedded input token into **query**, **key**, and **value** vectors
    """),
            mo.image(
                "notebooks/chapter3_attention/public/self_attention_key_value.png",
                height=350,
                width=700,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step-by-Step

    > Calculate for $x^2$
    """)
    return


@app.cell
def _(inputs):
    x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2
    return d_in, d_out, x_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here input dimension is 3 and output dimention is set to 2. In GPT-like models these are the same
    """)
    return


@app.cell
def _(d_in, d_out, torch):
    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    # setting requires_grad=False to simplify ouputs
    return W_key, W_query, W_value


@app.cell
def _(W_key, W_query, W_value, x_2):
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print(query_2)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
