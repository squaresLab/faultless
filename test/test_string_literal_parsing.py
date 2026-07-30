"""Test that string literals are parsed correctly.
"""

import unittest

from faultless.c import CHARACTER, parse_string_literal_content
from faultless.ir import CharLiteral, StringLiteral

NULL_BYTE = CharLiteral(0, CHARACTER)
class TestStringLiteralParsing(unittest.TestCase):
    def assertParsingEquivalent(self, literal: str, tokenized: list[str]):
        characters = [CharLiteral(ord(t), CHARACTER) for t in tokenized]
        characters.append(NULL_BYTE)
        self.assertListEqual(parse_string_literal_content(literal), characters)

    def test_simple_string(self):
        oracle = ["A", "B", "C"]
        self.assertParsingEquivalent("ABC", oracle)
    
    def test_newline(self):
        oracle = ["t", "w", "o", "\n", "l", "i", "n", "e", "s"]
        self.assertParsingEquivalent("two\\nlines", oracle)

    def test_escapes(self):
        oracle = ["\a", "\b", "\f", "\n", "\r", "\t", "\v", "\'", '\"', "\\", "?", "\0"]
        self.assertParsingEquivalent("\\a\\b\\f\\n\\r\\t\\v\\'\\\"\\\\\\?\\0", oracle)

    def test_octal_and_hex_notations(self):
        oracle = ["A", "1", "A", "G", "2"]
        self.assertParsingEquivalent("\\1011\\x41G2", oracle)

    def test_string_literal_hash_uses_character_values(self):
        left = StringLiteral("A", parse_string_literal_content("A"))
        right = StringLiteral("\\x41", parse_string_literal_content("\\x41"))

        self.assertEqual(left, right)
        self.assertEqual(hash(left), hash(right))
