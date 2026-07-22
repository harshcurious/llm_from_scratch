import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    return Path, mo


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib

    # Enable LaTeX rendering
    matplotlib.rcParams['text.usetex'] = True
    return (plt,)


@app.cell
def _():
    import ast
    import inspect


    def show(*args):
        """Prints arbitrary arguments alongside their variable names or expressions."""
        # 1. Get the frame of the caller
        frame = inspect.currentframe().f_back
        try:
            # 2. Extract the source context of the call
            frame_info = inspect.getframeinfo(frame)
            context = frame_info.code_context
            if not context:
                # Fallback if source code is unavailable (e.g., some REPLs)
                print(*args)
                return

            # 3. Parse the call statement using AST
            source_line = context[0].strip()
            tree = ast.parse(source_line)

            # Find the function call node
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Extract the string representations of the arguments
                    arg_names = [
                        ast.unparse(arg) if hasattr(ast, "unparse") else "arg"
                        for arg in node.args
                    ]

                    # 4. Map names to values and print
                    output = [
                        f"{name} = {repr(val)}" for name, val in zip(arg_names, args)
                    ]
                    print(", ".join(output))
                    return

        except Exception:
            # Fallback if AST parsing fails due to multi-line formatting or complex environments
            print(*args)
        finally:
            del frame  # Avoid reference cycles

    return (show,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Intro

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
    ### 1. The Why

    > **Long Seq Modeling problem**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
    Consider **text translation** model. 
    - Translating word-by-word => grammatical structures lost
    - Solution: DNN with encoder+decoder architecture ie **RNN**. (Roughly encoder reads original language; then decoder generates target language)
    - Issue: hidden state cannot be accessed at a later step! But size of hidden state is limited. Translating paragraphs/books loses structure. (Think about translating code from one language to another!!!)
    - Details in another notebook
    """),
        mo.mermaid(r"""
    graph LR

    S1[Sequential Input] --|encoder|--> S2[Hidden state] --|decoder|--> S3[Output]

    style S2 fill:#444400,stroke:#000
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Start of attentions mechanisms
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo):
    mo.md(rf"""
    #### Bahdanau attention

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
    ## 2. Simplified Self-Attention

    > Self-Attention without trainable weights
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **Self** in self-attention allows hidden states to look at _itself_ in the past
    - Assign weights (to the input) per position in input
    - Ends up capturing relationship between parts of input (eg words in sentences, pixels in an image)
    - Visually:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    Let $X = (x^1, x^2, ..., x^T)$ be the sentence "_Your journey starts with one step_". 

    Embedding:

    {mo.image("notebooks/chapter3_attention/public/self_attention_input.png", height=100, width=500)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""
    For $[x^2]$: 
    - assign weights $\alpha_{21}$, $\alpha_{22}$, $\alpha_{23}$, ..., $\alpha_{T}$ to all the input vectors
    - Via matric multiplications and additions we get the **context vector** $z^2$. 
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
    - enriched embedding of an input state using other input vectors
    - (like a given word enriched by the other elements of the sentence)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ### Step-by-Step Code for $x^{(2)}$ 
    {mo.image("notebooks/chapter3_attention/public/fixed_weight_context_vector.png", height=250, width=500)}
    """)
    return


@app.cell
def _():
    import torch

    # let's assume input embeddings
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
    # x^2
    query = inputs[1]

    # Generate attention scores (un-normalized \alpha) using dot product
    attn_scores_2 = torch.empty(inputs.shape[0])
    print(f"{query = }, {attn_scores_2 = }")
    for _i, _x_i in enumerate(inputs):
        attn_scores_2[_i] = torch.dot(_x_i, query)
    print(f"\n{attn_scores_2 = }")
    return attn_scores_2, query


@app.cell
def _(inputs, query):
    # Notice same as 
    inputs @ query
    return


@app.cell
def _(attn_scores_2):
    # Simple Normalization
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

    # Get \alpha after normalization
    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print("Attention weights:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())
    return


@app.cell
def _(attn_scores_2, torch):
    # Checking against in-built
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())
    return (attn_weights_2,)


@app.cell
def _(attn_weights_2, inputs, query, show, torch):
    # Calculating context vector z^2
    context_vec_2 = torch.zeros(query.shape)
    for _i, _x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[_i] * _x_i
    show(context_vec_2)
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
    ### Coding for the whole input

    Steps:
    1. Compute attentions scores
    2. Compute attention weights ($\mathbf{\alpha}$ matrix)
    3. Compute context vectors ($\mathbf{z}$ matrix)
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
    # step2: attention weights
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(attn_weights)
    return (attn_weights,)


@app.cell
def _(attn_weights, inputs):
    # Step3: context vector
    all_context_vecs = attn_weights @ inputs
    print(all_context_vecs)
    return (all_context_vecs,)


@app.cell(hide_code=True)
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
def _(attn_scores, inputs, plt):
    # Visualize the attention score matrix multiplication: inputs @ inputs.T
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 5))

    # Matrix 1: inputs (6x3)
    _ax1 = _axes[0]
    _im1 = _ax1.imshow(inputs.numpy(), cmap="Blues", aspect="auto")
    _ax1.set_title("Inputs X\n(6 tokens × 3 dims)", fontsize=12)
    _ax1.set_xlabel("Embedding dimension")
    _ax1.set_ylabel("Token position")
    _ax1.set_xticks(range(3))
    _ax1.set_yticks(range(6))
    _ax1.set_yticklabels([f"$x^{i + 1}$" for i in range(6)])
    # Annotate values on the result matrix
    for _i in range(6):
        for _j in range(3):
            _ax1.text(
                _j,
                _i,
                f"{inputs[_i, _j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if inputs[_i, _j] > 0.5 else "black",
            )

    # Matrix 2: inputs.T (3x6)
    _ax2 = _axes[1]
    _im2 = _ax2.imshow(inputs.T.numpy(), cmap="Oranges", aspect="auto")
    _ax2.set_title("Inputs$^\\top$\n(3 dims × 6 tokens)", fontsize=12)
    _ax2.set_xlabel("Token position")
    _ax2.set_ylabel("Embedding dimension")
    _ax2.set_xticks(range(6))
    _ax2.set_xticklabels([f"$x^{_i + 1}$" for _i in range(6)])
    for _i in range(3):
        for _j in range(6):
            _ax2.text(
                _j,
                _i,
                f"{inputs.T[_i, _j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if inputs.T[_i, _j] > 0.5 else "black",
            )

    # Result: attention scores (6x6)
    _ax3 = _axes[2]
    _im3 = _ax3.imshow(attn_scores.numpy(), cmap="Purples", aspect="auto")
    _ax3.set_title("Attention Scores\nX @ X$^\\top$ (6 × 6)", fontsize=12, usetex=False)
    _ax3.set_xlabel("Key token (j)")
    _ax3.set_ylabel("Query token (i)")
    _ax3.set_xticks(range(6))
    _ax3.set_xticklabels([f"$x^{_i + 1}$" for _i in range(6)])
    _ax3.set_yticks(range(6))
    _ax3.set_yticklabels([f"$x^{_i + 1}$" for _i in range(6)])

    # Annotate values on the result matrix
    for _i in range(6):
        for _j in range(6):
            _ax3.text(
                _j,
                _i,
                f"{attn_scores[_i, _j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if attn_scores[_i, _j] > 0.5 else "black",
            )

    for _ax in _axes:
        _ax.grid(False)

    plt.suptitle(
        "Matrix Multiplication: Attention Scores = Inputs @ Inputs$^\\top$",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _():
    _ax1.set_yticklabels([f"$x^{{{i + 1}}}$" for i in range(6)])
    return


@app.cell(hide_code=True)
def _(all_context_vecs, attn_scores, attn_weights, plt):
    # import matplotlib.pyplot as plt

    __fig, _axes = plt.subplots(1, 3, figsize=(16, 5))

    # Attention scores
    _ax1 = _axes[0]
    _im1 = _ax1.imshow(
        attn_scores.numpy(),
        cmap="Purples",
        aspect="equal",
    )
    _ax1.set_title("Step 1: Attention Scores\n(unnormalized)", fontsize=12)
    _ax1.set_xlabel("Key (j)")
    _ax1.set_ylabel("Query (i)")
    for _i in range(6):
        for _j in range(6):
            _ax1.text(
                _j,
                _i,
                f"{attn_scores[_i, _j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if attn_scores[_i, _j] > 0.5 else "black",
            )

    # Attention weights (softmax normalized)
    _ax2 = _axes[1]
    _im2 = _ax2.imshow(
        attn_weights.numpy(),
        cmap="YlOrRd",
        aspect="equal",
    )
    _ax2.set_title("Step 2: Attention Weights\n(softmax per row)", fontsize=12)
    _ax2.set_xlabel("Key (j)")
    _ax2.set_ylabel("Query (i)")
    for _i in range(6):
        for _j in range(6):
            _ax2.text(
                _j,
                _i,
                f"{attn_weights[_i, _j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    # Context vectors
    _ax3 = _axes[2]
    _im3 = _ax3.imshow(all_context_vecs.numpy(), cmap="Greens", aspect="auto")
    _ax3.set_title("Step 3: Context Vectors\n(weights @ inputs)", fontsize=12)
    _ax3.set_xlabel("Embedding dimension")
    _ax3.set_ylabel("Token position")
    for _i in range(6):
        for _j in range(3):
            _ax3.text(
                _j,
                _i,
                f"{all_context_vecs[_i, _j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    for ax in _axes:
        ax.set_yticks(range(6))
        ax.set_yticklabels([f"$x^{i + 1}$" for i in range(6)])

    _axes[0].set_xticks(range(6))
    _axes[0].set_xticklabels([f"$x^{j + 1}$" for j in range(6)])
    _axes[1].set_xticks(range(6))
    _axes[1].set_xticklabels([f"$x^{j + 1}$" for j in range(6)])
    _axes[2].set_xticks(range(3))

    __fig.suptitle(
        "Self-Attention Pipeline: X → Scores → Weights → Context Vectors",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Adding trainable weights

    True self-attention
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - We will implement the version used in early GPT models _scaled dot-product attention_
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""
    - 3 trainable weight matrices: $W_q$, $W_k$, and $W_v$
    - Projects embedded input token into **query**, **key**, and **value** vectors
        - borrows terminology from no-sql db
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

    > As last time, calculate for $x^2$
    """)
    return


@app.cell
def _(inputs):
    x_2 = inputs[1]
    d_in_2 = inputs.shape[1]
    d_out_2 = 2
    return d_in_2, d_out_2, x_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here input dimension is 3 and output dimention is set to 2. In GPT-like models these are equal (and much larger)
    """)
    return


@app.cell
def _(d_in_2, d_out_2, torch):
    # weight initialization
    torch.manual_seed(123)
    W_query_2 = torch.nn.Parameter(
        torch.rand(d_in_2, d_out_2), requires_grad=False
    )
    W_key_2 = torch.nn.Parameter(torch.rand(d_in_2, d_out_2), requires_grad=False)
    W_value_2 = torch.nn.Parameter(
        torch.rand(d_in_2, d_out_2), requires_grad=False
    )
    # setting requires_grad=False to simplify ouputs
    return


@app.cell
def _(W_key, W_query, W_value, show, x_2):
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    show(query_2, key_2, value_2)
    return (query_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We require the key & value vectors for all input elements as they are used for calculating attention weights
    """)
    return


@app.cell
def _(W_key, W_value, inputs):
    keys = inputs @ W_key
    values = inputs @ W_value
    print(f"{keys.shape = }")
    print(f"{values.shape = }")
    return keys, values


@app.cell(hide_code=True)
def _(W_key, inputs, keys, plt, values):
    _fig, _axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # Input matrix X (6x3)
    _ax = _axes[0]
    _ax.imshow(inputs.numpy(), cmap="Blues", aspect="auto")
    _ax.set_title("Inputs X (6×3)", fontsize=11)
    _ax.set_xlabel("d_in=3")
    _ax.set_ylabel("Tokens")
    _ax.set_yticks(range(6))
    _ax.set_yticklabels([f"$x^{i + 1}$" for i in range(6)])
    for _i in range(6):
        for _j in range(3):
            _ax.text(
                _j,
                _i,
                f"{inputs[_i, _j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if inputs[_i, _j] > 0.5 else "black",
            )

    # W_key matrix
    _ax = _axes[1]
    _ax.imshow(W_key.detach().numpy(), cmap="Reds", aspect="auto")
    _ax.set_title("W_key (3×2)", fontsize=11)
    _ax.set_xlabel("d_out=2")
    _ax.set_ylabel("d_in=3")
    for _i in range(3):
        for _j in range(2):
            _ax.text(
                _j,
                _i,
                f"{W_key[_i, _j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if W_key[_i, _j] > 0.5 else "black",
            )

    # Keys = inputs @ W_key
    _ax = _axes[2]
    _ax.imshow(keys.numpy(), cmap="Oranges", aspect="auto")
    _ax.set_title("Keys = X @ W_key (6×2)", fontsize=11, usetex=False)
    _ax.set_xlabel("d_out=2")
    _ax.set_ylabel("Tokens")
    _ax.set_yticks(range(6))
    _ax.set_yticklabels([f"$x^{i + 1}$" for i in range(6)])
    for _i in range(6):
        for _j in range(2):
            _ax.text(
                _j,
                _i,
                f"{keys[_i, _j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if keys[_i, _j] > 1 else "black",
            )

    # Values = inputs @ W_value
    _ax = _axes[3]
    _ax.imshow(values.numpy(), cmap="Greens", aspect="auto")
    _ax.set_title("Values = X @ W_value (6×2)", fontsize=11, usetex=False)
    _ax.set_xlabel("d_out=2")
    _ax.set_ylabel("Tokens")
    _ax.set_yticks(range(6))
    _ax.set_yticklabels([f"$x^{i + 1}$" for i in range(6)])
    for _i in range(6):
        for _j in range(2):
            _ax.text(
                _j,
                _i,
                f"{values[_i, _j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    for _ax in _axes:
        _ax.grid(False)

    plt.suptitle(
        "Trainable Weight Projections: X @ W → Keys & Values", fontsize=14, y=1.02, usetex=False
    )
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(keys, query_2, show):
    keys_2 = keys[1]
    attn_scores_22 = query_2.dot(keys_2)
    show(keys_2, attn_scores_22)
    return


@app.cell
def _(keys, query_2):
    attn_score_2 = query_2 @ keys.T
    print(attn_score_2)
    return


@app.cell
def _(attn_score_22, attn_weight_2, keys, torch):
    d_k = keys.shape[-1]
    attn_weight_22 = torch.softmax(attn_score_22 / d_k**0.5, dim=-1)
    print(attn_weight_2)
    return


@app.cell
def _(attn_weights_22, values):
    context_vec_22 = attn_weights_22 @ values
    print(context_vec_22)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
