import re


WORD_POS = 0


def load_glove(glove_path):
    word_index = dict()
    glove_matrix = []
    vocab = []

    with open(glove_path, 'r') as glove_file:
        for index, glove_line in enumerate(glove_file.readlines()):
            glove_features = glove_line.split()
            word_index[glove_features[WORD_POS]] = index + 1
            vocab.append(glove_features[WORD_POS])
            glove_matrix.append([float(value) for value in glove_features[1:]])

    return word_index, glove_matrix, vocab


def remove_html_from_text(text):
    return re.sub(r'<br\s/><br\s/>', ' ', text)


def remove_special_characters_from_text(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`;-]", ' ', text)
    return re.sub(r'\s{2,}', ' ', text)


def to_lower(review_text):
    return review_text.lower()
