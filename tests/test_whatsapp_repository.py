import unittest

from src.infrastructure.adapters.db.whatsapp_repository import _decode_settings


class DecodeSettingsTest(unittest.TestCase):
    def test_json_text_from_asyncpg_is_decoded(self) -> None:
        self.assertEqual(
            _decode_settings('{"custom_instructions": "oi", "quiz": null}'),
            {"custom_instructions": "oi", "quiz": None},
        )

    def test_already_decoded_dict_is_returned_as_is(self) -> None:
        self.assertEqual(_decode_settings({"quiz": None}), {"quiz": None})

    def test_missing_invalid_or_non_object_json_becomes_empty_dict(self) -> None:
        for raw in (None, "", "{invalido", "[1, 2]", "null", 42):
            with self.subTest(raw=raw):
                self.assertEqual(_decode_settings(raw), {})


if __name__ == "__main__":
    unittest.main()
