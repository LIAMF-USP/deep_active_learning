import re


def remove_html_from_text(text):
    return re.sub(r'<br\s/><br\s/>', ' ', text)


def remove_special_characters_from_text(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", ' ', text)
    return re.sub(r'\s{2,}', ' ', text)


def to_lower(review_text):
    return review_text.lower()
