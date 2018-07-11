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


def evaluate(model, dataloader, aid_to_ans):
    # switch to evaluate mode
    model.eval()
    # disable autograd tracking
    torch.set_grad_enabled(False)

    results = []

    for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        q_id = sample['question_id']
        q = sample['question']
        img = sample["image"]
        ans_label = sample['answer_id']
        ans_type = sample['answer_type']
        lengths = sample['question_len']

        q = q.cuda()
        img = img.cuda()
        ans_label = ans_label.cuda()

        output = model(img, q, lengths)

        # this is not needed but we want to be numerically stable
        output = functional.softmax(output, dim=1)

        ans = torch.max(output, dim=1)[1]

        for idx in range(ans.size(0)):
            results.append(
                {'question_id': q_id[idx].item(), 'answer': aid_to_ans[ans[idx].item()]})

    return results


def main():
    args = parse_args()

    # Set the GPU to use
    torch.cuda.set_device(args.gpu)

    vqa_loader = dataset.get_dataloader(osp.expanduser(args.annotations), osp.expanduser(args.questions),
                                        args.images, args, split="train", raw_images=args.raw_images, transforms=None)
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
                                        args.val_images, args, split="val", raw_images=args.raw_images,
                                        maps=maps, vocab=vocab, shuffle=False, transforms=val_transform)

    arch = Models[args.arch].value
    model = arch(len(vocab), output_dim=args.top_answer_limit,
                 raw_images=args.raw_images)

    if args.resume:
        state = torch.load(args.resume)
        model.load_state_dict(state["model"])

    else:
        print(
            "No trained model weights provided. Don't expect the answers to be meaningful.")

    if torch.cuda.is_available():
        model.cuda()

    with torch.no_grad():
        results = evaluate(model, val_loader, maps["aid_to_ans"])

    with open("VQA_OpenEnded_MSCOCO_results.json", 'w') as r:
        json.dump(results, r)


if __name__ == "__main__":
    main()
