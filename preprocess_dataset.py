import argparse

from preprocessing.dataset import MovieReviewDataset


def create_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d',
                        '--data-dir',
                        type=str,
                        help='The location of the Large Movie Review Dataset')

    parser.add_argument('-dod',
                        '--data-output-dir',
                        type=str,
                        help='The output dir for the formatted dataset')

    parser.add_argument('-trsp',
                        '--train-save-path',
                        type=str,
                        help='The location to save the formatted train dataset')

    parser.add_argument('-vsp',
                        '--validation-save-path',
                        type=str,
                        help='The location to save the formatted validation dataset')

    parser.add_argument('-tsp',
                        '--test-save-path',
                        type=str,
                        help='The location to save the formatted test dataset')

    parser.add_argument('-ef',
                        '--embedding-file',
                        type=str,
                        help='The location of the embedding file')

    parser.add_argument('-ep',
                        '--embedding-path',
                        type=str,
                        help='Location of the embedding file (Testing Dataset Only)')

    parser.add_argument('-ewi',
                        '--embedding-wordindex-path',
                        type=str,
                        help='Location of the embedding word index file (Testing Dataset Only)')

    parser.add_argument('-es',
                        '--embed-size',
                        type=int,
                        help='The embedding size of the embedding file')

    parser.add_argument('-s',
                        '--sentence-size',
                        type=int,
                        help=('The sentence size that will be used in the model.' +
                              'If a sentence in our dataset is larger than this variable' +
                              'It will be cropped to this size. Otherwise, it will be padded' +
                              'with an special character'))

    parser.add_argument('-o',
                        '--output-dir',
                        type=str,
                        help='The path of the new formatted dataset (TFRecord)')

    return parser


def main():
    parser = create_argument_parser()
    user_args = vars(parser.parse_args())

    train_save_path = user_args['train_save_path']
    validation_save_path = user_args['validation_save_path']
    test_save_path = user_args['test_save_path']

    data_dir = user_args['data_dir']
    data_output_dir = user_args['data_output_dir']
    output_dir = user_args['output_dir']

    embedding_file = user_args['embedding_file']
    embed_size = user_args['embed_size']
    embedding_path = user_args['embedding_path']
    embedding_wordindex_path = user_args['embedding_wordindex_path']
    sentence_size = user_args['sentence_size']

    movie_review_dataset = MovieReviewDataset(
        train_save_path=train_save_path,
        validation_save_path=validation_save_path,
        test_save_path=test_save_path,
        data_dir=data_dir,
        data_output_dir=data_output_dir,
        output_dir=output_dir,
        embedding_file=embedding_file,
        embed_size=embed_size,
        embedding_path=embedding_path,
        embedding_wordindex_path=embedding_wordindex_path,
        sentence_size=sentence_size)
    movie_review_dataset.create_dataset()


if __name__ == '__main__':
    main()
