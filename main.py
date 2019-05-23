import os.path as osp

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms

import dataset
import trainer
import visualize
from arguments import parse_args
from models.model import Models


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda:{0}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    vqa_loader = dataset.get_dataloader(osp.expanduser(args.annotations),
                                        osp.expanduser(args.questions),
                                        args.images, args, split='train',
                                        raw_images=args.raw_images,
                                        transforms=transform)
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
                                        args.val_images, args, split='val',
                                        raw_images=args.raw_images,
                                        maps=maps, vocab=vocab,
                                        shuffle=False, transforms=val_transform)

    arch = Models[args.arch].value
    model = arch(len(vocab), image_dim=args.image_dim,
                 output_dim=args.top_answer_limit, raw_images=args.raw_images)

    if args.resume:
        state = torch.load(args.resume)
        model.load_state_dict(state["model"])

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval,
                                    gamma=args.lr_decay)

    if args.visualize:
        vis = visualize.Visualizer(args.port)
    else:
        vis = None

    print("Beginning training")
    print("#"*80)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        trainer.train(model, vqa_loader, criterion,
                      optimizer, epoch, args, device, vis=vis)
        # trainer.evaluate(model, val_loader, criterion, epoch, args, device, vis=vis)

    print("Training complete!")


if __name__ == "__main__":
    main()
