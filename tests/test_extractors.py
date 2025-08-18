from scraper.extractors import extract_buttons, extract_images_from_result_media


def test_extract_buttons_basic():
    html = """
    <html><body>
    <button id="b1">Click me</button>
    <input type="submit" id="submit1" value="Submit Form" />
    </body></html>
    """
    buttons = extract_buttons(html)
    texts = sorted([b["text"] for b in buttons])
    assert "Click me" in texts
    assert "Submit Form" in texts


def test_extract_images_from_result_media():
    media = {
        "images": [
            {"src": "https://example.com/a.png", "alt": "A"},
            {"src": "https://example.com/b.jpg", "alt": ""},
        ]
    }
    images = extract_images_from_result_media(media)
    assert any(i["src"].endswith("a.png") for i in images)
    assert any(i["src"].endswith("b.jpg") for i in images)


