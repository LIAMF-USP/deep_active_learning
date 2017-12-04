import matplotlib
matplotlib.use('Agg')  # noqa

import matplotlib.pyplot as plt


def accuracy_graph(train_accuracy, validation_accuracy, save_path):
    line1, = plt.plot(train_accuracy, label='Train')
    line2, = plt.plot(validation_accuracy, label='Validation')

    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title('Train vs Validation accuracy')
    plt.legend(handles=[line1, line2], loc='lower right')

    plt.savefig(save_path)
