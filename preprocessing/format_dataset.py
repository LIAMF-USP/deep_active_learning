import re


def remove_html_from_text(review_text):
    return re.sub(r'<br\s/><br\s/>', ' ', review_text)


def to_lower(review_text):
    return review_text.lower()
