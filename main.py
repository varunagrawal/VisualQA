import os.path as osp
import argparse
import dataset
from models.model import Models
import trainer
import visualize
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("annotations", metavar="TRAIN_ANN", help="The VQA annotations JSON file")
    parser.add_argument("questions", metavar="TRAIN_QUES", help="The VQA questions JSON file")
    parser.add_argument("images", metavar="TRAIN_IMAGES",
                        help="The file containing torch tensors of the FC7 embeddings of COCO images")
    parser.add_argument("val_annotations", metavar="VAL_ANN", help="The VQA val annotations JSON file")
    parser.add_argument("val_questions", metavar="VAL_QUES", help="The VQA val questions JSON file")
    parser.add_argument("val_images", metavar="VAL_IMAGES",
                        help="The file containing torch tensors of the FC7 embeddings of val COCO images")
    parser.add_argument("--embed_question", action="store_true",
                        help="Return the question as a list of word IDs so we can use an embedding layer on it")
    parser.add_argument("--top_answer_limit", default=1000, help="The number of answers to consider as viable options")
    parser.add_argument("--max_length", default=25, help="The maximum length to consider to each question")
    parser.add_argument("--arch", default="DeeperLSTM", help="The model to use for VQA",
                        choices=tuple([name for name, _ in Models.__members__.items()]))
    parser.add_argument("--epochs", default=150, help="Number of training epochs")
    parser.add_argument("--start_epoch", default=0, help="Starting epoch")
    parser.add_argument('--lr', "--learning_rate", default=3e-4, help="Learning rate")
    parser.add_argument("--lr_decay", default=0.95, help="The learning rate decay")
    parser.add_argument("--decay_interval", default=1, help="The epoch step size at which to decay the learning rate")
    parser.add_argument("--betas", default=(0.8, 0.999), nargs="+", type=float)
    parser.add_argument("--weight-decay", '--wd', default=0.0, help="Optimizer weight decay")
    parser.add_argument("--print-freq", default=100, help="How frequently to print training stats")
    parser.add_argument("--batch_size", default=16, help="Batch Size")
    parser.add_argument('-j', "--num_workers", default=4)
    parser.add_argument("--save_dir", default="weights", help="Directory where model weights are stored")
    parser.add_argument("--gpu", default=-0, help="The GPU to use for training. For multi-gpu setups.")
    parser.add_argument("--visualize_freq", default=20,
                        help="Number of iterations after which Visdom should update loss graph")
    parser.add_argument("--port", default=8097, help="The port for the Visdom server")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Set the GPU to use
    torch.cuda.set_device(args.gpu)

    annotations = osp.expanduser(args.annotations)
    questions = osp.expanduser(args.questions)

    vqa_loader = dataset.get_train_dataloader(annotations, questions, args.images, args)
    # We always use the vocab from the training set
    vocab = vqa_loader.dataset.vocab

    maps = {
        "word_to_wid": vqa_loader.dataset.word_to_wid,
        "wid_to_word": vqa_loader.dataset.wid_to_word,
        "ans_to_aid": vqa_loader.dataset.ans_to_aid,
        "aid_to_ans": vqa_loader.dataset.aid_to_ans,
    }
    val_loader = dataset.get_val_dataloader(osp.expanduser(args.val_annotations),
                                            osp.expanduser(args.val_questions),
                                            args.val_images, args,
                                            maps=maps, vocab=vocab, shuffle=False)

    arch = Models[args.arch].value
    model = arch(len(vocab), output_dim=args.top_answer_limit)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.lr_decay)

    vis = visualize.Visualizer(args.port)

    print("Beginning training")
    print("#"*80)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        trainer.train(model, vqa_loader, criterion, optimizer, epoch, args, vis=vis)
        trainer.evaluate(model, val_loader, criterion, epoch, args, vis=vis)

        break
    print("Training complete!")


if __name__ == "__main__":
    main()
