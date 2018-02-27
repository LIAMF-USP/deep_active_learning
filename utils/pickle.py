import pickle


def load(pkl_file):
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)


def save(save_data, pkl_file):
    with open(pkl_file, 'wb') as f:
        pickle.dump(save_data, f)
