import numpy as np


def variation_ratio(mc_counts):
    """
    This function receives a list of tuples of V elements, where V is the number
    of examples predicted by the model. Each tuple has the following format:

    ([3, 2], 0), where the first argument is count how many 0 and 1 labels were generated, and the
    second argument, the label which was generated the most.

    Furthermore, this function will return an V X 1 array containing the variation ratio
    for each example in the all_preds matrix.
    """

    variation_ratios = []
    for bincounts, label in mc_counts:
        variation_ratio = 1 - (bincounts[label] / sum(bincounts))
        variation_ratios.append(variation_ratio)

    return variation_ratios


def entropy(predictions, num_samples):
    average_predictions = np.divide(predictions, num_samples)

    np.seterr(divide='ignore')
    log_average_predictions = np.log2(average_predictions)
    np.seterr(divide='warn')

    entropy_average = -np.multiply(average_predictions, log_average_predictions)
    entropy_final_value = np.sum(entropy_average, axis=1)

    return entropy_final_value


def bald(predictions, all_entropy_dropout, num_samples):
    entropy_value = entropy(predictions, num_samples)
    average_entropy = np.divide(all_entropy_dropout, num_samples)

    bald = entropy_value - average_entropy

    return bald
