from __future__ import annotations

import unittest

from agent_platform.radar_routes import is_radar_callback


class RadarRoutesTest(unittest.TestCase):
    def test_legacy_radar_callbacks_are_claimed(self) -> None:
        callbacks = (
            "research:home",
            "research:auto",
            "research:accounts",
            "research:import",
            "research:youtube",
            "research:instagram",
            "research:results",
            "scripts:list",
            "auto:youtube",
            "auto:instagram",
            "research_confirm:RS-20260807-005",
            "research_content_retry:RS-20260807-005",
            "research_cancel:RS-20260807-005",
            "research_run_results:RS-20260807-005",
            "research_run:RS-20260807-005",
            "result_script:42",
            "result_handoff:42",
            "result:42",
            "radar_shared_item:CR-20260807-001",
        )

        for callback in callbacks:
            with self.subTest(callback=callback):
                self.assertTrue(is_radar_callback(callback))

    def test_unrelated_partner_callbacks_are_not_claimed(self) -> None:
        callbacks = (
            "partner:script",
            "image:new",
            "image_confirm:IMG-1",
            "list:start",
            "list_finish:DL-1",
            "shared:add",
            "shared_handoff:CR-1",
        )

        for callback in callbacks:
            with self.subTest(callback=callback):
                self.assertFalse(is_radar_callback(callback))


if __name__ == "__main__":
    unittest.main()
