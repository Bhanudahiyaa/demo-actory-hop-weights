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


def extract_images_from_result_media(media: Dict[str, Any]) -> List[Dict[str, str]]:
    images = []
    for img in (media or {}).get("images", []):
        images.append({"src": img.get("src", ""), "alt": img.get("alt", "")})
    return images


