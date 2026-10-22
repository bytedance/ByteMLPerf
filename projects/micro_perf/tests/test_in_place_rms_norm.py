"""Regression tests for in-place LLM RMSNorm base operations."""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as functional


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "projects" / "micro_perf"))

from op_defs.llm_ops.head_rms_norm import HeadRMSNormOp  # noqa: E402
from op_defs.llm_ops.qk_rms_norm import QKRMSNormOp  # noqa: E402


class InPlaceRMSNormTest(unittest.TestCase):
    def test_head_rms_norm_updates_only_selected_heads(self):
        token_data = torch.tensor(
            [[[1.0, 2.0, 3.0], [2.0, 4.0, 8.0], [3.0, 5.0, 7.0]]]
        )
        expected = token_data.clone()
        weight = torch.ones(3)
        expected[:, 1:2, :] = functional.rms_norm(
            expected[:, 1:2, :], [3], weight=weight, eps=1e-5
        )

        op = HeadRMSNormOp.__new__(HeadRMSNormOp)
        op.norm_head_start = 1
        op.norm_head_end = 2
        op.eps = 1e-5

        result = op.vendor_impl_run({"token_data": token_data, "norm_weight": weight})

        self.assertIs(result, token_data)
        torch.testing.assert_close(result, expected)

    def test_qk_rms_norm_updates_q_and_k_but_preserves_v(self):
        token_data = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            ]
        )
        expected = token_data.clone()
        q_weight = torch.ones(2)
        k_weight = torch.ones(2)
        expected[:, :4] = functional.rms_norm(
            expected[:, :4].view(-1, 2, 2), [2], weight=q_weight, eps=1e-5
        ).view(-1, 4)
        expected[:, 4:6] = functional.rms_norm(
            expected[:, 4:6].view(-1, 1, 2), [2], weight=k_weight, eps=1e-5
        ).view(-1, 2)

        op = QKRMSNormOp.__new__(QKRMSNormOp)
        op.q_head_num = 2
        op.kv_head_num = 1
        op.qk_head_dim = 2
        op.v_head_dim = 3
        op.eps = 1e-5

        result = op.vendor_impl_run(
            {
                "token_data": token_data,
                "q_norm_weight": q_weight,
                "k_norm_weight": k_weight,
            }
        )

        self.assertIs(result, token_data)
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
