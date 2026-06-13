import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Use <https://d2l.ai/chapter_recurrent-neural-networks/index.html> for reference
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch

    return mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vanilla RNN Forward Pass

    We will implement the recurrent update directly with PyTorch layers, without using `torch.nn.RNN`.

    At each time step:

    - $h_{t}^{\text{pre}} = W_{xh} x_t + W_{hh} h_{t-1} + b_h$
    - $h_t = \tanh(h_{t}^{\text{pre}})$
    - $y_t = W_{hy} h_t + b_y$

    We will run this over a short sequence with batch size 1 and inspect every intermediate tensor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        flowchart LR
            xt["x_t<br/>current input"] --> pre["pre_activation<br/>W_xh x_t + W_hh h_(t-1) + b_h"]
            hprev["h_(t-1)<br/>previous hidden state"] --> pre
            pre --> tanh["tanh"]
            tanh --> ht["h_t<br/>new hidden state"]
            ht --> yt["y_t<br/>output"]
            ht -.-> next["used as h_(t-1)<br/>at next step"]
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import anywidget
    import traitlets

    return anywidget, traitlets


@app.cell
def _(torch):
    torch.manual_seed(7)

    sequence = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [0.0, 1.0],
        ]
    )

    _, input_size = sequence.shape
    hidden_size = 3
    output_size = 2

    print("sequence shape:", tuple(sequence.shape))
    print(sequence)
    return hidden_size, input_size, output_size, sequence


@app.cell
def _(hidden_size, input_size, output_size, torch):
    x_to_h = torch.nn.Linear(input_size, hidden_size, bias=False)
    h_to_h = torch.nn.Linear(hidden_size, hidden_size, bias=True)
    h_to_y = torch.nn.Linear(hidden_size, output_size, bias=True)

    h_0 = None
    if h_0 is None:
        initial_h = torch.zeros(hidden_size)
    else:
        initial_h = h_0.clone()

    print("initial hidden state h_0:")
    print(initial_h)
    print("-" * 10)
    print(f"{x_to_h = }")
    print(x_to_h.weight)
    print("-" * 10)
    print(f"{h_to_h = }")
    print(h_to_h.weight)
    print(f"{h_to_h.bias = }")
    print("-" * 10)
    print(f"{h_to_y = }")
    print(h_to_y.weight)
    print(f"{h_to_y.bias = }")
    return h_to_h, h_to_y, initial_h, x_to_h


@app.cell
def _(h_to_h, h_to_y, initial_h, sequence, torch, x_to_h):
    hidden_states = []
    outputs = []
    step_summaries = []
    current_h = initial_h

    for t, x_t in enumerate(sequence):
        pre_activation = x_to_h(x_t) + h_to_h(current_h)
        current_h = torch.tanh(pre_activation)
        y_t = h_to_y(current_h)

        hidden_states.append(current_h)
        outputs.append(y_t)
        step_summaries.append(
            {
                "t": t,
                "x_t": x_t,
                "pre_activation": pre_activation,
                "h_t": current_h,
                "y_t": y_t,
            }
        )

    for summary in step_summaries:
        print(f"t = {summary['t']}")
        print("x_t =", summary["x_t"])
        print("pre_activation =", summary["pre_activation"])
        print("h_t =", summary["h_t"])
        print("y_t =", summary["y_t"])
        print()
    return hidden_states, outputs, step_summaries


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class RNNStepWidget(anywidget.AnyWidget):
        _esm = """
        function formatVector(values) {
          if (!Array.isArray(values)) {
            return "[]";
          }
          return `[${values.map((value) => Number(value).toFixed(4)).join(", ")}]`;
        }

        function render({ model, el }) {
          el.replaceChildren();

          const root = document.createElement("div");
          root.className = "rnn-step-widget";

          const title = document.createElement("div");
          title.className = "rnn-step-widget__title";
          title.textContent = "Interactive RNN step explorer";

          const subtitle = document.createElement("div");
          subtitle.className = "rnn-step-widget__subtitle";
          subtitle.textContent = "Scrub across time to inspect the input, pre-activation, hidden state, and output.";

          const controls = document.createElement("div");
          controls.className = "rnn-step-widget__controls";

          const prevButton = document.createElement("button");
          prevButton.textContent = "Previous";

          const slider = document.createElement("input");
          slider.type = "range";
          slider.min = "0";
          slider.step = "1";

          const nextButton = document.createElement("button");
          nextButton.textContent = "Next";

          const stepLabel = document.createElement("div");
          stepLabel.className = "rnn-step-widget__step-label";

          controls.append(prevButton, slider, nextButton, stepLabel);

          const grid = document.createElement("div");
          grid.className = "rnn-step-widget__grid";

          const fieldSpecs = [
            ["input", "x_t", "Current input vector"],
            ["pre_activation", "pre-activation", "W_xh x_t + W_hh h_(t-1) + b_h"],
            ["hidden", "h_t", "Hidden state after tanh"],
            ["output", "y_t", "Output projection from h_t"],
          ];

          const cards = new Map();
          for (const [key, label, caption] of fieldSpecs) {
            const card = document.createElement("div");
            card.className = "rnn-step-widget__card";

            const cardLabel = document.createElement("div");
            cardLabel.className = "rnn-step-widget__card-label";
            cardLabel.textContent = label;

            const cardCaption = document.createElement("div");
            cardCaption.className = "rnn-step-widget__card-caption";
            cardCaption.textContent = caption;

            const cardValue = document.createElement("pre");
            cardValue.className = "rnn-step-widget__card-value";

            card.append(cardLabel, cardCaption, cardValue);
            grid.append(card);
            cards.set(key, cardValue);
          }

          root.append(title, subtitle, controls, grid);
          el.append(root);

          const getSteps = () => model.get("steps") || [];
          const clampStep = (step) => {
            const maxStep = Math.max(0, getSteps().length - 1);
            return Math.max(0, Math.min(step, maxStep));
          };

          const saveStep = (step) => {
            model.set("step", clampStep(Number(step)));
            model.save_changes();
          };

          prevButton.addEventListener("click", () => saveStep((model.get("step") || 0) - 1));
          nextButton.addEventListener("click", () => saveStep((model.get("step") || 0) + 1));
          slider.addEventListener("input", () => saveStep(slider.value));

          const redraw = () => {
            const steps = getSteps();
            if (steps.length === 0) {
              slider.disabled = true;
              prevButton.disabled = true;
              nextButton.disabled = true;
              stepLabel.textContent = "No RNN steps available";
              for (const valueEl of cards.values()) {
                valueEl.textContent = "[]";
              }
              return;
            }

            const step = clampStep(model.get("step") || 0);
            slider.disabled = false;
            slider.max = String(steps.length - 1);
            slider.value = String(step);
            prevButton.disabled = step === 0;
            nextButton.disabled = step === steps.length - 1;
            stepLabel.textContent = `t = ${step} / ${steps.length - 1}`;

            const current = steps[step];
            for (const [key, valueEl] of cards.entries()) {
              valueEl.textContent = formatVector(current[key]);
            }
          };

          model.on("change:step", redraw);
          model.on("change:steps", redraw);
          redraw();
        }

        export default { render };
        """

        _css = """
        .rnn-step-widget {
          border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
          border-radius: 18px;
          padding: 1rem;
          background: linear-gradient(180deg, rgba(56, 189, 248, 0.08), rgba(99, 102, 241, 0.05));
          display: grid;
          gap: 0.9rem;
          font-family: Inter, ui-sans-serif, system-ui, sans-serif;
        }

        .rnn-step-widget__title {
          font-size: 1rem;
          font-weight: 700;
        }

        .rnn-step-widget__subtitle {
          font-size: 0.92rem;
          opacity: 0.78;
          line-height: 1.4;
        }

        .rnn-step-widget__controls {
          display: grid;
          grid-template-columns: auto minmax(140px, 1fr) auto auto;
          gap: 0.75rem;
          align-items: center;
        }

        .rnn-step-widget__controls button {
          border: 1px solid color-mix(in srgb, currentColor 14%, transparent);
          background: rgba(255, 255, 255, 0.72);
          color: inherit;
          border-radius: 999px;
          padding: 0.45rem 0.8rem;
          cursor: pointer;
          font: inherit;
        }

        .rnn-step-widget__controls button:disabled {
          opacity: 0.45;
          cursor: default;
        }

        .rnn-step-widget__step-label {
          font-size: 0.9rem;
          font-weight: 600;
          white-space: nowrap;
        }

        .rnn-step-widget__grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 0.75rem;
        }

        .rnn-step-widget__card {
          border: 1px solid color-mix(in srgb, currentColor 12%, transparent);
          border-radius: 14px;
          padding: 0.85rem;
          background: rgba(255, 255, 255, 0.74);
          display: grid;
          gap: 0.35rem;
        }

        .rnn-step-widget__card-label {
          font-size: 0.92rem;
          font-weight: 700;
        }

        .rnn-step-widget__card-caption {
          font-size: 0.8rem;
          opacity: 0.72;
          min-height: 2.2em;
        }

        .rnn-step-widget__card-value {
          margin: 0;
          padding: 0.7rem;
          border-radius: 10px;
          background: rgba(15, 23, 42, 0.08);
          overflow-x: auto;
          font-size: 0.88rem;
          line-height: 1.5;
        }

        @media (prefers-color-scheme: dark) {
          .rnn-step-widget {
            background: linear-gradient(180deg, rgba(14, 165, 233, 0.15), rgba(79, 70, 229, 0.12));
          }

          .rnn-step-widget__controls button,
          .rnn-step-widget__card {
            background: rgba(15, 23, 42, 0.65);
          }

          .rnn-step-widget__card-value {
            background: rgba(15, 23, 42, 0.9);
          }
        }
        """

        steps = traitlets.List().tag(sync=True)
        step = traitlets.Int(0).tag(sync=True)

    return (RNNStepWidget,)


@app.cell(hide_code=True)
def _(RNNStepWidget, mo, step_summaries):
    steps = [
        {
            "input": summary["x_t"].detach().tolist(),
            "pre_activation": summary["pre_activation"].detach().tolist(),
            "hidden": summary["h_t"].detach().tolist(),
            "output": summary["y_t"].detach().tolist(),
        }
        for summary in step_summaries
    ]

    step_widget = mo.ui.anywidget(RNNStepWidget(steps=steps))
    step_widget
    return


@app.cell
def _(hidden_states, outputs, torch):
    hidden_states_tensor = torch.stack(hidden_states)
    outputs_tensor = torch.stack(outputs)

    print("all hidden states shape:", tuple(hidden_states_tensor.shape))
    print(hidden_states_tensor)
    print()
    print("all outputs shape:", tuple(outputs_tensor.shape))
    print(outputs_tensor)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
