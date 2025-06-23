import unittest
from utils import convert_markdown_to_docs_format

class TestMarkdownFormatting(unittest.TestCase):
    def assert_style_present(self, requests, style_type, expected_value):
        """Helper to check if a specific style is present in the requests."""
        for req in requests:
            if style_type in req:
                if style_type == "updateParagraphStyle":
                    if req[style_type]["paragraphStyle"]["namedStyleType"] == expected_value:
                        return True
                elif style_type == "updateTextStyle":
                    style = req[style_type]["textStyle"]
                    if any(style.get(k) == v for k, v in expected_value.items()):
                        return True
                elif style_type == "createParagraphBullets":
                    if req[style_type]["bulletPreset"] == expected_value:
                        return True
        return False

    def test_headings(self):
        """Test all heading levels."""
        markdown = """# Heading 1
## Heading 2
### Heading 3
#### Heading 4"""
        
        requests = convert_markdown_to_docs_format(markdown)
        
        self.assertTrue(self.assert_style_present(requests, "updateParagraphStyle", "HEADING_1"))
        self.assertTrue(self.assert_style_present(requests, "updateParagraphStyle", "HEADING_2"))
        self.assertTrue(self.assert_style_present(requests, "updateParagraphStyle", "HEADING_3"))
        self.assertTrue(self.assert_style_present(requests, "updateParagraphStyle", "HEADING_4"))

    def test_bold_and_italic(self):
        """Test bold and italic text."""
        markdown = """This is **bold** text
This is *italic* text
This is **bold and *italic* mixed**"""
        
        requests = convert_markdown_to_docs_format(markdown)
        
        self.assertTrue(self.assert_style_present(requests, "updateTextStyle", {"bold": True}))
        self.assertTrue(self.assert_style_present(requests, "updateTextStyle", {"italic": True}))

    def test_tables(self):
        """Test table formatting."""
        markdown = """| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| **Bold** | *Italic* |"""
        
        requests = convert_markdown_to_docs_format(markdown)
        
        # Check for monospace font
        found_monospace = False
        for req in requests:
            if "updateTextStyle" in req:
                style = req["updateTextStyle"]["textStyle"]
                if "weightedFontFamily" in style and style["weightedFontFamily"]["fontFamily"] == "Courier New":
                    found_monospace = True
                    break
        self.assertTrue(found_monospace, "Table should use monospace font")
        
        # Check that separator row is removed
        text_content = [req["insertText"]["text"] for req in requests if "insertText" in req]
        separator_pattern = r"-+"
        self.assertTrue(all(not line.strip().startswith(separator_pattern) for line in text_content))

    def test_bullets_and_nesting(self):
        """Test bullet points and nesting levels."""
        markdown = """- Level 1
  - Level 2
    - Level 3
- Back to Level 1"""
        
        requests = convert_markdown_to_docs_format(markdown)
        
        # Check for bullet formatting
        self.assertTrue(self.assert_style_present(requests, "createParagraphBullets", "BULLET_DISC_CIRCLE_SQUARE"))
        
        # Check nesting levels
        nesting_levels = []
        for req in requests:
            if "createParagraphBullets" in req:
                nesting_levels.append(req["createParagraphBullets"].get("nestingLevel", 0))
        
        self.assertEqual(nesting_levels, [0, 1, 2, 0], "Incorrect nesting levels")

    def test_horizontal_rules(self):
        """Test horizontal rules."""
        markdown = """Before rule
---
After rule"""
        
        requests = convert_markdown_to_docs_format(markdown)
        
        # Check for horizontal rule
        found_rule = False
        for req in requests:
            if "insertText" in req:
                if req["insertText"]["text"].strip() == "_" * 40:
                    found_rule = True
                    break
        self.assertTrue(found_rule, "Horizontal rule not found")

    def test_mixed_formatting(self):
        """Test multiple formatting features together."""
        markdown = """# Main Title
This is a **bold** and *italic* test.

| Table | Test |
|-------|------|
| **Bold** | *Italic* |

- Bullet 1
  - Nested with **bold**
    - Deep nested with *italic*

---

## Section with everything"""
        
        requests = convert_markdown_to_docs_format(markdown)
        
        # Check all formatting types are present
        self.assertTrue(self.assert_style_present(requests, "updateParagraphStyle", "HEADING_1"))
        self.assertTrue(self.assert_style_present(requests, "updateTextStyle", {"bold": True}))
        self.assertTrue(self.assert_style_present(requests, "updateTextStyle", {"italic": True}))
        self.assertTrue(self.assert_style_present(requests, "createParagraphBullets", "BULLET_DISC_CIRCLE_SQUARE"))
        
        # Check for table and horizontal rule
        found_monospace = False
        found_rule = False
        for req in requests:
            if "updateTextStyle" in req:
                style = req["updateTextStyle"]["textStyle"]
                if "weightedFontFamily" in style and style["weightedFontFamily"]["fontFamily"] == "Courier New":
                    found_monospace = True
            if "insertText" in req and req["insertText"]["text"].strip() == "_" * 40:
                found_rule = True
        
        self.assertTrue(found_monospace, "Table formatting not found")
        self.assertTrue(found_rule, "Horizontal rule not found")

if __name__ == '__main__':
    unittest.main() 