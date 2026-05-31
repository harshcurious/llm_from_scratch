import pathlib
import sys
import unittest


NOTEBOOK_DIR = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "chapter2_text_data_prep"
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from tiktoken_educational import bpe_encode_steps, bpe_train_steps, visualise_tokens


class TestTiktokenEducationalAnimations(unittest.TestCase):
    def test_bpe_encode_steps_capture_each_merge(self):
        mergeable_ranks = {
            b"a": 0,
            b"b": 1,
            b"c": 2,
            b"ab": 3,
            b"bc": 4,
            b"abc": 5,
        }

        steps = bpe_encode_steps(mergeable_ranks, b"abc")

        self.assertEqual(steps, [[b"a", b"b", b"c"], [b"ab", b"c"], [b"abc"]])

    def test_bpe_train_steps_capture_visualisation_frames(self):
        ranks, frames, captions = bpe_train_steps("aa aa", vocab_size=257, pat_str=r"\S+")

        self.assertIn(b"aa", ranks)
        self.assertEqual(frames, [[b"aa", b"aa"]])
        self.assertEqual(len(captions), 1)
        self.assertIn("b'a' + b'a'", captions[0])

    def test_visualise_tokens_returns_animation_html(self):
        html = visualise_tokens(
            [[b"a", b"b"], [b"ab"]],
            captions=["step 1", "step 2"],
            frame_interval_ms=2400,
            title="Demo animation",
        )

        self.assertIn("data-frame-interval=\"2400\"", html)
        self.assertIn("Demo animation", html)
        self.assertIn("step 1", html)
        self.assertIn("ab", html)


if __name__ == "__main__":
    unittest.main()
