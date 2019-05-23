"""Command line arguments."""

import argparse
from models import Models


def parse_args():
    """Create and parse the command line arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument("annotations", metavar="TRAIN_ANN",
                        help="The VQA annotations JSON file")

    parser.add_argument("questions", metavar="TRAIN_QUES",
                        help="The VQA questions JSON file")
    parser.add_argument("--images", metavar="TRAIN_IMAGES",
                        help="The file containing either the tensors of the CNN embeddings of \
                        COCO images or the json of images")
    parser.add_argument("val_annotations", metavar="VAL_ANN",
                        help="The VQA val annotations JSON file")
    parser.add_argument("val_questions", metavar="VAL_QUES",
                        help="The VQA val questions JSON file")
    parser.add_argument("--val_images", metavar="VAL_IMAGES",
                        help="The file containing either the tensors of the CNN embeddings or the \
                        json file of COCO images")
    parser.add_argument("--image_root",
                        help="Root path to the images directory")

    parser.add_argument("--raw_images", action="store_true", default=False,
                        help="Flag to indicate if we're using the the raw images instead of the \
                        preprocessed embeddings")
    parser.add_argument("--image_dim", type=int, default=1024,
                        help="Dimension of image features extracted from \
                        a feature extractor convnet")
    parser.add_argument("--embed_question", action="store_true",
                        help="Return the question as a list of word IDs so we can use \
                        an embedding layer on it")
    parser.add_argument("--top_answer_limit", default=1000, type=int,
                        help="The number of answers to consider as viable options")
    parser.add_argument("--max_length", default=26, type=int,
                        help="The maximum length to consider to each question")
    parser.add_argument("--img_size", default=224, type=int,
                        help="The size of the image to pass to the CNN")

    parser.add_argument("--arch", default="DeeperLSTM", help="The model to use for VQA",
                        choices=tuple([name for name, _ in Models.__members__.items()]))
    parser.add_argument("--epochs", default=200, type=int,
                        help="Number of training epochs")
    parser.add_argument("--start_epoch", default=0,
                        type=int, help="Starting epoch")
    parser.add_argument('--lr', "--learning_rate",
                        default=3e-4, type=float, help="Learning rate")
    parser.add_argument("--lr_decay", default=0.99997592083,
                        type=float, help="The learning rate decay")
    parser.add_argument("--decay_interval", default=10, type=int,
                        help="The epoch step size at which to decay the learning rate")
    parser.add_argument("--betas", default=(0.8, 0.999), nargs="+", type=float)
    parser.add_argument("--weight-decay", '--wd', default=0.0,
                        type=float, help="Optimizer weight decay")
    parser.add_argument("--print-freq", default=100, type=int,
                        help="How frequently to print training stats")

    parser.add_argument("--batch_size", default=16,
                        help="Batch Size", type=int)
    parser.add_argument('-j', "--num_workers", default=8, type=int)
    parser.add_argument("--save_dir", default="weights",
                        help="Directory where model weights are stored")
    parser.add_argument("--gpu", default=0, type=int,
                        help="The GPU to use for training. For multi-gpu setups.")

    parser.add_argument("--resume", default=None,
                        help="Path to the weights file to resume training from")
    parser.add_argument("--results_file",
                        default="VQA_OpenEnded_COCO_results.json")

    parser.add_argument("--visualize", action="store_true",
                        help="Flag to indicate use of Visdom for visualization")
    parser.add_argument("--visualize_freq", default=20, type=int,
                        help="Number of iterations after which Visdom should update loss graph")
    parser.add_argument("--port", default=8097, type=int,
                        help="The port for the Visdom server")

    args = parser.parse_args()
    return args
