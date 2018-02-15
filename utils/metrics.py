
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
