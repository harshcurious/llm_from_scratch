"""This is an educational implementation of the byte pair encoding algorithm."""

from __future__ import annotations

import collections
import html
import json
import uuid

import regex

import tiktoken


ANIMATION_FRAME_INTERVAL_MS = 2500
TOKEN_BACKGROUND = ["#d35d6e", "#d2a84a", "#a9b665", "#4fb286", "#4d96ff", "#7d6bff", "#b565d9"]


class SimpleBytePairEncoding:
    def __init__(self, *, pat_str: str, mergeable_ranks: dict[bytes, int]) -> None:
        """Creates an Encoding object."""
        # A regex pattern string that is used to split the input text
        self.pat_str = pat_str
        # A dictionary mapping token bytes to their ranks. The ranks correspond to merge priority
        self.mergeable_ranks = mergeable_ranks

        self._decoder = {token: token_bytes for token_bytes, token in mergeable_ranks.items()}
        self._pat = regex.compile(pat_str)

    def encode(self, text: str, visualise: str | None = "colour") -> list[int]:
        """Encodes a string into tokens.

        >>> enc.encode("hello world")
        [388, 372]
        """
        # Use the regex to split the text into (approximately) words
        words = self._pat.findall(text)
        tokens = []
        for word in words:
            # Turn each word into tokens, using the byte pair encoding algorithm
            word_bytes = word.encode("utf-8")
            word_tokens = bpe_encode(self.mergeable_ranks, word_bytes, visualise=visualise)
            tokens.extend(word_tokens)
        return tokens

    def encode_steps(self, text: str) -> list[dict[str, str | list[list[bytes]] | list[str]]]:
        words = self._pat.findall(text)
        word_steps = []
        for word in words:
            steps = bpe_encode_steps(self.mergeable_ranks, word.encode("utf-8"))
            word_steps.append(
                {
                    "word": word,
                    "steps": steps,
                    "captions": build_bpe_encode_captions(steps),
                }
            )
        return word_steps

    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decodes a list of tokens into bytes.

        >>> enc.decode_bytes([388, 372])
        b'hello world'
        """
        return b"".join(self._decoder[token] for token in tokens)

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of tokens into a string.

        Decoded bytes are not guaranteed to be valid UTF-8. In that case, we replace
        the invalid bytes with the replacement character "�".

        >>> enc.decode([388, 372])
        'hello world'
        """
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")

    def decode_tokens_bytes(self, tokens: list[int]) -> list[bytes]:
        """Decodes a list of tokens into a list of bytes.

        Useful for visualising how a string is tokenised.

        >>> enc.decode_tokens_bytes([388, 372])
        [b'hello', b' world']
        """
        return [self._decoder[token] for token in tokens]

    @staticmethod
    def train(
        training_data: str,
        vocab_size: int,
        pat_str: str,
        visualise: str | None = "colour",
    ):
        """Train a BPE tokeniser on some data!"""
        mergeable_ranks = bpe_train(
            data=training_data,
            vocab_size=vocab_size,
            pat_str=pat_str,
            visualise=visualise,
        )
        return SimpleBytePairEncoding(pat_str=pat_str, mergeable_ranks=mergeable_ranks)

    @staticmethod
    def from_tiktoken(encoding):
        if isinstance(encoding, str):
            encoding = tiktoken.get_encoding(encoding)
        return SimpleBytePairEncoding(
            pat_str=encoding._pat_str, mergeable_ranks=encoding._mergeable_ranks
        )


def bpe_encode(
    mergeable_ranks: dict[bytes, int], input: bytes, visualise: str | None = "colour"
) -> list[int]:
    steps = bpe_encode_steps(mergeable_ranks, input)
    if visualise:
        if visualise in ["colour", "color", "animation"]:
            visualise_tokens(
                steps,
                captions=build_bpe_encode_captions(steps),
                title=f"BPE encoding for {input!r}",
            )
        elif visualise == "simple":
            for parts in steps:
                print(parts)
            print()

    tokens = [mergeable_ranks[part] for part in steps[-1]]
    return tokens


def bpe_encode_steps(mergeable_ranks: dict[bytes, int], input: bytes) -> list[list[bytes]]:
    parts = [bytes([b]) for b in input]
    steps = [parts.copy()]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        if min_rank is None:
            break
        assert min_idx is not None

        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
        steps.append(parts.copy())
    return steps


def bpe_train(
    data: str, vocab_size: int, pat_str: str, visualise: str | None = "colour"
) -> dict[bytes, int]:
    ranks, frames, captions = _run_bpe_training(
        data=data,
        vocab_size=vocab_size,
        pat_str=pat_str,
        collect_visualisation=bool(visualise),
    )

    if visualise:
        if visualise in ["colour", "color", "animation"]:
            visualise_tokens(
                frames,
                captions=captions,
                title="BPE training progression (first 50 words)",
            )
        elif visualise == "simple":
            for caption, frame in zip(captions, frames, strict=False):
                print(caption)
                print(frame)
                print()

    return ranks


def bpe_train_steps(
    data: str,
    vocab_size: int,
    pat_str: str,
) -> tuple[dict[bytes, int], list[list[bytes]], list[str]]:
    return _run_bpe_training(
        data=data,
        vocab_size=vocab_size,
        pat_str=pat_str,
        collect_visualisation=True,
    )


def _run_bpe_training(
    data: str,
    vocab_size: int,
    pat_str: str,
    *,
    collect_visualisation: bool,
) -> tuple[dict[bytes, int], list[list[bytes]], list[str]]:
    # First, add tokens for each individual byte value
    if vocab_size < 2**8:
        raise ValueError("vocab_size must be at least 256, so we can encode all bytes")
    ranks = {}
    for i in range(2**8):
        ranks[bytes([i])] = i

    # Splinter up our data into lists of bytes
    # data = "Hello world"
    # words = [
    #     [b'H', b'e', b'l', b'l', b'o'],
    #     [b' ', b'w', b'o', b'r', b'l', b'd']
    # ]
    words: list[list[bytes]] = [
        [bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, data)
    ]

    frames: list[list[bytes]] = []
    captions: list[str] = []

    # Now, use our data to figure out which merges we should make
    while len(ranks) < vocab_size:
        # Find the most common pair. This will become our next token
        stats = collections.Counter()
        for piece in words:
            for pair in zip(piece[:-1], piece[1:]):
                stats[pair] += 1

        most_common_pair = max(stats, key=lambda x: stats[x])
        token_bytes = most_common_pair[0] + most_common_pair[1]
        token = len(ranks)
        # Add the new token!
        ranks[token_bytes] = token

        # Now merge that most common pair in all the words. That is, update our training data
        # to reflect our decision to make that pair into a new token.
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == most_common_pair:
                    # We found our pair! Merge it
                    new_word.append(token_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if i == len(word) - 1:
                new_word.append(word[i])
            new_words.append(new_word)
        words = new_words

        if collect_visualisation:
            frames.append([token for word in words[:50] for token in word])
            captions.append(
                f"Merge step {len(captions) + 1}: {most_common_pair[0]!r} + {most_common_pair[1]!r} -> {token_bytes!r}"
            )

    return ranks, frames, captions


def build_bpe_encode_captions(steps: list[list[bytes]]) -> list[str]:
    captions = ["Initial split into single-byte tokens"]
    for step_number, (previous, current) in enumerate(zip(steps[:-1], steps[1:]), start=1):
        merge_index = _find_merge_index(previous, current)
        if merge_index is None:
            captions.append(f"Merge step {step_number}")
            continue
        merged = current[merge_index]
        captions.append(
            f"Merge step {step_number}: {previous[merge_index]!r} + {previous[merge_index + 1]!r} -> {merged!r}"
        )
    return captions


def _find_merge_index(previous: list[bytes], current: list[bytes]) -> int | None:
    for idx in range(len(current)):
        if idx + 1 >= len(previous):
            break
        if previous[idx] + previous[idx + 1] != current[idx]:
            continue
        if previous[:idx] == current[:idx] and previous[idx + 2 :] == current[idx + 1 :]:
            return idx
    return None


def visualise_tokens(
    token_values: list[bytes] | list[list[bytes]],
    *,
    captions: list[str] | None = None,
    frame_interval_ms: int = ANIMATION_FRAME_INTERVAL_MS,
    title: str | None = None,
) -> str:
    frames = _normalise_visualisation_frames(token_values)
    if not frames:
        raise ValueError("token_values must contain at least one frame")

    captions = captions or [f"Step {index + 1}" for index in range(len(frames))]
    if len(captions) != len(frames):
        raise ValueError("captions must match the number of frames")

    payload = [
        {
            "caption": caption,
            "tokens_html": _render_token_frame(frame),
        }
        for caption, frame in zip(captions, frames, strict=False)
    ]

    animation_id = f"bpe-animation-{uuid.uuid4().hex}"
    title_html = f'<div class="bpe-animation-title">{html.escape(title)}</div>' if title else ""
    animation_html = f"""
<div id="{animation_id}" class="bpe-animation-root" data-frame-interval="{frame_interval_ms}">
  <style>
    #{animation_id} {{
      background: #11110b;
      border: 1px solid #2b2b22;
      border-radius: 16px;
      color: #f4f4ef;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace;
      padding: 1rem;
      max-width: 100%;
      box-sizing: border-box;
    }}
    #{animation_id} .bpe-animation-title {{
      font-size: 1rem;
      font-weight: 700;
      margin-bottom: 0.75rem;
    }}
    #{animation_id} .bpe-animation-meta {{
      color: #c9c7be;
      font-size: 0.9rem;
      margin-bottom: 0.75rem;
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      flex-wrap: wrap;
    }}
    #{animation_id} .bpe-animation-stage {{
      align-items: flex-start;
      background: #171711;
      border-radius: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 0.25rem;
      min-height: 4.5rem;
      padding: 0.75rem;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    #{animation_id} .bpe-token {{
      border-radius: 8px;
      color: #101010;
      display: inline-block;
      line-height: 1.6;
      padding: 0.15rem 0.4rem;
    }}
  </style>
  {title_html}
  <div class="bpe-animation-meta">
    <div class="bpe-animation-caption"></div>
    <div class="bpe-animation-progress"></div>
  </div>
  <div class="bpe-animation-stage"></div>
</div>
<script>
(() => {{
  const root = document.getElementById({json.dumps(animation_id)});
  if (!root) return;

  const frames = {json.dumps(payload)};
  const stage = root.querySelector('.bpe-animation-stage');
  const caption = root.querySelector('.bpe-animation-caption');
  const progress = root.querySelector('.bpe-animation-progress');
  const intervalMs = Number(root.dataset.frameInterval || {frame_interval_ms});
  let frameIndex = 0;

  const render = () => {{
    const frame = frames[frameIndex];
    stage.innerHTML = frame.tokens_html;
    caption.textContent = frame.caption;
    progress.textContent = `Frame ${{frameIndex + 1}} / ${{frames.length}}`;
  }};

  render();
  if (frames.length <= 1) return;

  window.setInterval(() => {{
    frameIndex = (frameIndex + 1) % frames.length;
    render();
  }}, intervalMs);
}})();
</script>
""".strip()

    try:
        from IPython.display import HTML, display

        display(HTML(animation_html))
    except Exception:
        pass

    return animation_html


def _normalise_visualisation_frames(
    token_values: list[bytes] | list[list[bytes]],
) -> list[list[bytes]]:
    if not token_values:
        return []
    if isinstance(token_values[0], bytes):
        return [token_values]  # type: ignore[list-item]
    return token_values  # type: ignore[return-value]


def _render_token_frame(token_values: list[bytes]) -> str:
    unicode_token_values = [x.decode("utf-8", errors="replace") for x in token_values]

    running_length = 0
    last_color = None
    rendered_tokens = []
    for token in unicode_token_values:
        color = TOKEN_BACKGROUND[running_length % len(TOKEN_BACKGROUND)]
        if color == last_color:
            color = TOKEN_BACKGROUND[(running_length + 1) % len(TOKEN_BACKGROUND)]
        last_color = color
        running_length += len(token)
        token_text = token if token else "�"
        rendered_tokens.append(
            f'<span class="bpe-token" style="background:{color}">{html.escape(token_text)}</span>'
        )
    return "".join(rendered_tokens)


def train_simple_encoding():
    gpt2_pattern = (
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    with open(__file__) as f:
        data = f.read()

    enc = SimpleBytePairEncoding.train(data, vocab_size=600, pat_str=gpt2_pattern)

    print("This is the sequence of merges performed in order to encode 'hello world':")
    tokens = enc.encode("hello world")
    assert enc.decode(tokens) == "hello world"
    assert enc.decode_bytes(tokens) == b"hello world"
    assert enc.decode_tokens_bytes(tokens) == [b"hello", b" world"]

    return enc
