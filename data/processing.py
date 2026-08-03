import re
import unicodedata

def processing_text(text: str):
    text = unicodedata.normalize("NFC", text)
    text = text.lower()

    # Remove speaker label
    text = re.sub(
        r'^\s*[a-z][a-z .\'-]{0,80}:\s*',
        '',
        text
    )

    # Keep Vietnamese + English + digits + punctuation
    text = re.sub(
        r"[^a-z0-9"
        r"àáảãạăắằẳẵặâấầẩẫậ"
        r"đ"
        r"èéẻẽẹêếềểễệ"
        r"ìíỉĩị"
        r"òóỏõọôốồổỗộơớờởỡợ"
        r"ùúủũụưứừửữự"
        r"ỳýỷỹỵ"
        r"\s,.?!']",
        "",
        text
    )

    # Collapse spaces
    text = re.sub(r"\s+", " ", text)

    # Collapse repeated punctuation
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r",{2,}", ",", text)

    return text.strip()