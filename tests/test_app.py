"""Integration checks for Role 4's app assembly."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app
from providers import MockProvider


class ScriptedProvider:
    def __init__(self, replies: list[str]):
        self.replies = iter(replies)

    def generate(self, *_args, **_kwargs) -> str:
        return next(self.replies)


class AppIntegrationTests(unittest.TestCase):
    def test_grouped_test_cases_are_flattened(self):
        cases = app.load_test_cases()
        self.assertEqual(len(cases), 8)
        self.assertEqual(cases[0]["id"], "S01")
        self.assertIn("category", cases[0])

    def test_react_executes_registry_tool_then_returns_answer(self):
        provider = ScriptedProvider(
            [
                'Thought: Cần xem thông tin căn.\nAction: {"tool": "get_property_details", "args": {"property_id": "CH001"}}',
                "Thought: Đã có dữ liệu.\nFinal Answer: Căn CH001 đã được xác minh.",
            ]
        )
        answer, trace = app.run_react_agent("Thông tin CH001", provider, verbose=False)

        self.assertEqual(answer, "Căn CH001 đã được xác minh.")
        self.assertEqual(len(trace), 1)
        self.assertIn("CH001", trace[0].observation)

    def test_booking_is_blocked_without_explicit_confirmation(self):
        result = app._execute_action(
            "create_booking",
            {
                "property_id": "CH001",
                "viewing_time": "2026-08-01 09:00",
                "customer_name": "An",
                "confirmed": True,
            },
            "Đặt lịch giúp tôi",
        )
        self.assertIn("AN TOÀN", result)

    def test_unknown_tool_is_blocked(self):
        self.assertIn("không được phép", app._execute_action("delete_everything", {}, "test"))

    def test_mock_baseline_can_hold_a_basic_conversation(self):
        answer = app.run_baseline_chatbot("xin chào", MockProvider(), verbose=False)
        self.assertIn("Chào bạn", answer)

    def test_mock_react_reads_budget_and_location(self):
        answer, _ = app.run_react_agent("Tìm nhà ở Quận 10 dưới 5 triệu", MockProvider(), verbose=False)
        self.assertIn("PT003", answer)


if __name__ == "__main__":
    unittest.main()
