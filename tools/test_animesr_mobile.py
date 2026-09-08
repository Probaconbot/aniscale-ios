"""Ensure mobile recurrence matches the training/inference network."""
import unittest

import torch
from torch.nn import functional as F

from cloud.animesr import AnimeSR
from tools.convert_animesr_mobile import AnimeSRCell


class MobileParityTest(unittest.TestCase):
    def test_feedback_channel_order(self):
        feedback = torch.arange(3 * 32 * 48).reshape(1, 3, 32, 48)
        torch.testing.assert_close(
            AnimeSRCell.pixel_unshuffle_x4(feedback),
            F.pixel_unshuffle(feedback, 4), rtol=0, atol=0,
        )

    def test_three_recurrent_frames(self):
        torch.manual_seed(91)
        torch.set_num_threads(2)
        reference = AnimeSR().eval()
        mobile = AnimeSRCell().eval()
        mobile.load_state_dict(reference.state_dict(), strict=True)
        feedback = torch.rand(1, 3, 32, 48)
        state = torch.rand(1, 64, 8, 12)
        mobile_feedback, mobile_state = feedback.clone(), state.clone()
        with torch.inference_mode():
            for _ in range(3):
                frames = torch.rand(1, 9, 8, 12)
                feedback, state = reference(frames, feedback, state)
                mobile_feedback, mobile_state = mobile(frames, mobile_feedback, mobile_state)
                torch.testing.assert_close(mobile_feedback, feedback, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(mobile_state, state, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
