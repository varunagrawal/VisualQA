import os.path as osp
import argparse
import dataset


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
    parser.add_argument("--top_answer_limit", default=1000, help="The number of answers to consider as viable options")
    parser.add_argument("--max_length", default=25, help="The maximum length to consider to each question")
    parser.add_argument("--batch_size", default=32)
    parser.add_argument('-j', "--num_workers", default=4)

    args = parser.parse_args()
    return args


args = parse_args()
annotations = osp.expanduser(args.annotations)
questions = osp.expanduser(args.questions)

vqa_loader = dataset.get_dataloader(annotations, questions, args.images, "train", args)

print("Beginning training")
for idx, sample in enumerate(vqa_loader):
    q = sample['question'].transpose(0, 1)  # convert to TimexBatch
    ans_label = sample['answer']
    break

print("Training complete!")
