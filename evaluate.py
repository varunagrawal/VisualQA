import os.path as osp
import dataset
from models.model import Models
import numpy as np
import torch
from torch.nn import functional
from torchvision import transforms
from arguments import parse_args
from tqdm import tqdm
import json


@torch.no_grad()
def evaluate(model, dataloader, aid_to_ans, device):
    # move to device and switch to evaluate mode
    model = model.to(device).eval()

    results = []

    for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        q_id = sample['question_id']
        q = sample['question']
        img = sample["image"]
        ans_type = sample['answer_type']
        lengths = sample['question_len']

        q = q.to(device)
        img = img.to(device)

        output = model(img, q, lengths)

        ans = torch.max(output.cpu(), dim=1)[1]

        results.append(
            {
                'question_id': q_id.item(),
                'answer': aid_to_ans[ans.item()]
            }
        )

    return results


def main():
    args = parse_args()

    # Set the GPU to use
    torch.cuda.set_device(args.gpu)

    vqa_loader = dataset.get_dataloader(osp.expanduser(args.annotations),
                                        osp.expanduser(args.questions),
                                        args.images, args,
                                        split="train", raw_images=args.raw_images, transforms=None)
    # We always use the vocab from the training set
    vocab = vqa_loader.dataset.vocab

    maps = {
        "vocab": vocab,
        "word_to_wid": vqa_loader.dataset.word_to_wid,
        "wid_to_word": vqa_loader.dataset.wid_to_word,
        "ans_to_aid": vqa_loader.dataset.ans_to_aid,
        "aid_to_ans": vqa_loader.dataset.aid_to_ans,
    }
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_loader = dataset.get_dataloader(osp.expanduser(args.val_annotations),
                                        osp.expanduser(args.val_questions),
                                        args.val_images, args,
                                        split="val", raw_images=args.raw_images,
                                        maps=maps, vocab=vocab,
                                        shuffle=False, transforms=val_transform)

    arch = Models[args.arch].value
    model = arch(len(vocab), image_dim=args.image_dim,
                 output_dim=args.top_answer_limit, raw_images=args.raw_images)

    if args.resume:
        state = torch.load(args.resume)
        model.load_state_dict(state["model"])

    else:
        print("No trained model weights provided. Don't expect the answers to be meaningful.")

    if torch.cuda.is_available():
        device = torch.device('cuda:{0}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    results = evaluate(model, val_loader, maps["aid_to_ans"], device)

    with open(args.results_file, 'w') as r:
        json.dump(results, r)

    print("Results saved to {0}".format(args.results_file))


if __name__ == "__main__":
    main()
