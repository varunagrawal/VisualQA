import os.path as osp
import argparse
import dataset
from models.model import DeeperLSTM
import trainer
import torch
from torch import optim
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotations", metavar="ANN",
                        default="~/datasets/VQA2/v2_mscoco_train2014_annotations.json",
                        help="The VQA annotations JSON file")
    parser.add_argument("--questions", metavar="QUES",
                        default="~/datasets/VQA2/v2_OpenEnded_mscoco_train2014_questions.json",
                        help="The VQA questions JSON file")
    parser.add_argument("--images", default="coco_train_vgg_fc7.pth",
                        help="The file containing torch tensors of the FC7 embeddings of COCO images")
    parser.add_argument("--val-annotations", metavar="ANN",
                        default="~/datasets/VQA2/v2_mscoco_val2014_annotations.json",
                        help="The VQA val annotations JSON file")
    parser.add_argument("--val-questions", metavar="QUES",
                        default="~/datasets/VQA2/v2_OpenEnded_mscoco_val2014_questions.json",
                        help="The VQA val questions JSON file")
    parser.add_argument("--val-images", default="coco_val_vgg_fc7.pth",
                        help="The file containing torch tensors of the FC7 embeddings of val COCO images")
    parser.add_argument("--embed_question", action="store_true")
    parser.add_argument("--top_answer_limit", default=1000, help="The number of answers to consider as viable options")
    parser.add_argument("--max_length", default=25, help="The maximum length to consider to each question")
    parser.add_argument("--epochs", default=50, help="Number of training epochs")
    parser.add_argument("--start_epoch", default=0, help="Starting epoch")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", '--wd', default=0.0005, help="Optimizer weight decay")
    parser.add_argument("--print-freq", default=100, help="How frequently to print training stats")
    parser.add_argument("--batch_size", default=32)
    parser.add_argument('-j', "--num_workers", default=4)
    parser.add_argument("--save_dir", default="weights", help="Directory where model weights are stored")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    annotations = osp.expanduser(args.annotations)
    questions = osp.expanduser(args.questions)

    vqa_loader = dataset.get_dataloader(annotations, questions, args.images, "train", args)

    val_loader = dataset.get_dataloader(osp.expanduser(args.val_annotations),
                                        osp.expanduser(args.val_questions),
                                        args.val_images,
                                        "val", args)

    vocab = vqa_loader.dataset.vocab
    model = DeeperLSTM(len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Beginning training")
    print("#"*80)

    for epoch in range(args.start_epoch, args.epochs):
        trainer.train(model, vqa_loader, criterion, optimizer, epoch, args)
        trainer.evaluate(model, val_loader, criterion, epoch, args)

    print("Training complete!")


if __name__ == "__main__":
    main()
