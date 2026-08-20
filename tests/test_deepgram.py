from __future__ import annotations

import unittest

from agent_platform.deepgram import parse_deepgram_response


class DeepgramParsingTest(unittest.TestCase):
    def test_parse_response_extracts_transcript_metadata(self) -> None:
        transcript = parse_deepgram_response(
            {
                "metadata": {"duration": 12.5},
                "results": {
                    "channels": [
                        {
                            "alternatives": [
                                {
                                    "transcript": "Привет, это тест.",
                                    "confidence": 0.98,
                                }
                            ]
                        }
                    ]
                },
            }
        )

        self.assertEqual(transcript.text, "Привет, это тест.")
        self.assertEqual(transcript.confidence, 0.98)
        self.assertEqual(transcript.duration_seconds, 12.5)


if __name__ == "__main__":
    unittest.main()
