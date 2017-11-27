from datetime import datetime


def create_unique_name(base_name):
    """
    This function is used to create a unique name for saving tensorboard
    information.

    It adds to base_name received as parameter the current data and time when
    a model is created.

    Args:
        base_name: A string containing the base name where to save tensorboard
                   information
    Returns:
        unique_name: A unique name that represents the directory name where
                     tensorboard information will be saved.
    """

    date_str = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    return base_name + '-' + date_str
