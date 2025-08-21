from bs4 import BeautifulSoup
from typing import List, Dict, Any


def extract_buttons(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html or "", "lxml")
    buttons: List[Dict[str, Any]] = []
    for el in soup.find_all("button"):
        buttons.append(
            {
                "text": el.get_text(strip=True) or "",
                "id": el.get("id", ""),
                "name": el.get("name", ""),
                "type": el.get("type", ""),
            }
        )
    # input type=button/submit
    for el in soup.find_all("input", {"type": ["button", "submit"]}):
        buttons.append(
            {
                "text": el.get("value", ""),
                "id": el.get("id", ""),
                "name": el.get("name", ""),
                "type": el.get("type", ""),
            }
        )
    return buttons


def extract_input_fields(html: str) -> List[Dict[str, Any]]:
    """Extract all input fields including text, email, password, etc."""
    soup = BeautifulSoup(html or "", "lxml")
    inputs = []
    
    for el in soup.find_all("input"):
        inputs.append({
            "id": el.get("id", ""),
            "name": el.get("name", ""),
            "type": el.get("type", "text"),
            "placeholder": el.get("placeholder", ""),
            "required": el.get("required") is not None,
            "value": el.get("value", ""),
        })
    
    return inputs


def extract_textareas(html: str) -> List[Dict[str, Any]]:
    """Extract all textarea elements."""
    soup = BeautifulSoup(html or "", "lxml")
    textareas = []
    
    for el in soup.find_all("textarea"):
        textareas.append({
            "id": el.get("id", ""),
            "name": el.get("name", ""),
            "placeholder": el.get("placeholder", ""),
            "rows": el.get("rows", ""),
            "cols": el.get("cols", ""),
            "required": el.get("required") is not None,
        })
    
    return textareas


def extract_links(html: str) -> List[Dict[str, Any]]:
    """Extract all link elements with enhanced metadata."""
    soup = BeautifulSoup(html or "", "lxml")
    links = []
    
    for el in soup.find_all("a"):
        links.append({
            "href": el.get("href", ""),
            "text": el.get_text(strip=True) or "",
            "title": el.get("title", ""),
            "id": el.get("id", ""),
            "class": " ".join(el.get("class", [])),
            "target": el.get("target", ""),
            "rel": el.get("rel", ""),
        })
    
    return links


def extract_images(html: str) -> List[Dict[str, Any]]:
    """Extract all image elements with enhanced metadata."""
    soup = BeautifulSoup(html or "", "lxml")
    images = []
    
    for el in soup.find_all("img"):
        images.append({
            "src": el.get("src", ""),
            "alt": el.get("alt", ""),
            "id": el.get("id", ""),
            "class": " ".join(el.get("class", [])),
            "width": el.get("width", ""),
            "height": el.get("height", ""),
            "title": el.get("title", ""),
        })
    
    return images


def extract_stylesheets(html: str) -> List[Dict[str, Any]]:
    """Extract all stylesheet resources."""
    soup = BeautifulSoup(html or "", "lxml")
    stylesheets = []
    
    # External stylesheets
    for el in soup.find_all("link", rel="stylesheet"):
        stylesheets.append({
            "url": el.get("href", ""),
            "type": "stylesheet",
            "media": el.get("media", ""),
            "integrity": el.get("integrity", ""),
            "crossorigin": el.get("crossorigin", ""),
        })
    
    # Inline styles
    for el in soup.find_all("style"):
        stylesheets.append({
            "url": "inline",
            "type": "inline_style",
            "content": el.get_text()[:200] + "..." if len(el.get_text()) > 200 else el.get_text(),
        })
    
    return stylesheets


def extract_scripts(html: str) -> List[Dict[str, Any]]:
    """Extract all script resources."""
    soup = BeautifulSoup(html or "", "lxml")
    scripts = []
    
    for el in soup.find_all("script"):
        src = el.get("src", "")
        if src:
            # External script
            scripts.append({
                "url": src,
                "type": "external_script",
                "async": el.get("async") is not None,
                "defer": el.get("defer") is not None,
                "integrity": el.get("integrity", ""),
                "crossorigin": el.get("crossorigin", ""),
            })
        else:
            # Inline script
            scripts.append({
                "url": "inline",
                "type": "inline_script",
                "content": el.get_text()[:200] + "..." if len(el.get_text()) > 200 else el.get_text(),
            })
    
    return scripts


def extract_fonts(html: str) -> List[Dict[str, Any]]:
    """Extract font resources from stylesheets and font-face declarations."""
    soup = BeautifulSoup(html or "", "lxml")
    fonts = []
    
    # Font links
    for el in soup.find_all("link", rel="preload", as_="font"):
        fonts.append({
            "url": el.get("href", ""),
            "type": "font",
            "format": el.get("type", ""),
            "crossorigin": el.get("crossorigin", ""),
        })
    
    # Look for font-face in style tags
    for style in soup.find_all("style"):
        style_text = style.get_text()
        if "@font-face" in style_text:
            # Simple extraction of font URLs from CSS
            import re
            font_urls = re.findall(r'url\(["\']?([^"\')\s]+)["\']?\)', style_text)
            for url in font_urls:
                fonts.append({
                    "url": url,
                    "type": "font",
                    "format": "css_declared",
                    "crossorigin": "",
                })
    
    return fonts


def extract_forms(html: str) -> List[Dict[str, Any]]:
    """Extract form elements with their structure."""
    soup = BeautifulSoup(html or "", "lxml")
    forms = []
    
    for el in soup.find_all("form"):
        form_data = {
            "id": el.get("id", ""),
            "name": el.get("name", ""),
            "action": el.get("action", ""),
            "method": el.get("method", "get"),
            "enctype": el.get("enctype", ""),
            "target": el.get("target", ""),
            "inputs": [],
            "buttons": [],
        }
        
        # Extract form inputs
        for input_el in el.find_all("input"):
            form_data["inputs"].append({
                "type": input_el.get("type", "text"),
                "name": input_el.get("name", ""),
                "id": input_el.get("id", ""),
                "required": input_el.get("required") is not None,
            })
        
        # Extract form buttons
        for button_el in el.find_all("button"):
            form_data["buttons"].append({
                "type": button_el.get("type", "submit"),
                "text": button_el.get_text(strip=True) or "",
                "name": button_el.get("name", ""),
            })
        
        forms.append(form_data)
    
    return forms


def extract_media_resources(html: str) -> List[Dict[str, Any]]:
    """Extract audio, video, and other media resources."""
    soup = BeautifulSoup(html or "", "lxml")
    media = []
    
    # Audio elements
    for el in soup.find_all("audio"):
        media.append({
            "url": el.get("src", ""),
            "type": "audio",
            "controls": el.get("controls") is not None,
            "autoplay": el.get("autoplay") is not None,
            "loop": el.get("loop") is not None,
        })
    
    # Video elements
    for el in soup.find_all("video"):
        media.append({
            "url": el.get("src", ""),
            "type": "video",
            "width": el.get("width", ""),
            "height": el.get("height", ""),
            "controls": el.get("controls") is not None,
            "autoplay": el.get("autoplay") is not None,
        })
    
    # Source elements (for audio/video)
    for el in soup.find_all("source"):
        media.append({
            "url": el.get("src", ""),
            "type": el.get("type", "media"),
            "media": el.get("media", ""),
        })
    
    return media


def extract_images_from_result_media(media: Dict[str, Any]) -> List[Dict[str, str]]:
    images = []
    for img in (media or {}).get("images", []):
        images.append({"src": img.get("src", ""), "alt": img.get("alt", "")})
    return images


def extract_all_dom_features(html: str) -> Dict[str, List[Dict[str, Any]]]:
    """Extract all DOM features in one comprehensive function."""
    return {
        "buttons": extract_buttons(html),
        "inputs": extract_input_fields(html),
        "textareas": extract_textareas(html),
        "links": extract_links(html),
        "images": extract_images(html),
        "stylesheets": extract_stylesheets(html),
        "scripts": extract_scripts(html),
        "fonts": extract_fonts(html),
        "forms": extract_forms(html),
        "media": extract_media_resources(html),
    }


