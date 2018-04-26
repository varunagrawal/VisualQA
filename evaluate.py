import os.path as osp
import dataset
from models.model import Models
import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torchvision import transforms
from arguments import parse_args
from tqdm import tqdm


def evaluate(model, dataloader):
    # switch to evaluate mode
    model.eval()

    results = {
        'yes/no': np.zeros(len(dataloader.dataset)),
        'number': np.zeros(len(dataloader.dataset)),
        'other': np.zeros(len(dataloader.dataset))
    }

    for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        q = sample['question']
        img = sample["image"]
        ans_label = sample['answer_id']
        ans_type = sample['answer_type']

        q = q.cuda()
        img = img.cuda()
        ans_label = ans_label.cuda()

        output = model(img, q)
        ans = torch.zeros(img.size(0)).long().cuda()#torch.max(functional.softmax(output, dim=1), dim=1)[1]
        result = ans.eq(ans_label).cpu().detach().numpy()

        for idx, (r, a) in enumerate(zip(result, ans_type)):
            results[a][i*dataloader.batch_size + idx] = result[idx]

    return results


def main():
    args = parse_args()

    # Set the GPU to use
    torch.cuda.set_device(args.gpu)

    vqa_loader = dataset.get_train_dataloader(osp.expanduser(args.annotations),
                                              osp.expanduser(args.questions),
                                              args.images, args, raw_images=args.raw_images,
                                              transforms=None)
    # We always use the vocab from the training set
    vocab = vqa_loader.dataset.vocab

    maps = {
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
    val_loader = dataset.get_val_dataloader(osp.expanduser(args.val_annotations),
                                            osp.expanduser(args.val_questions),
                                            args.val_images, args, raw_images=args.raw_images,
                                            maps=maps, vocab=vocab, shuffle=False, transforms=val_transform)

    arch = Models[args.arch].value
    model = arch(len(vocab), output_dim=args.top_answer_limit, raw_images=args.raw_images)

    if args.resume:
        state = torch.load(args.resume)
        model.load_state_dict(state["model"])

    else:
        print("No trained model weights provided. Don't expect the answers to be meaningful.")

    if torch.cuda.is_available():
        model.cuda()

    with torch.no_grad():
        results = evaluate(model, val_loader)

    for k in results.keys():
        acc = results[k].sum() / results[k].shape
        print("Accuracy for {0} type answers: \t\t{1}".format(k, acc))


if __name__ == "__main__":
    main()
