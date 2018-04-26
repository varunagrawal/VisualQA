import os.path as osp
import dataset
from models.model import Models
import trainer
import visualize
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch import nn
from torchvision import transforms
from arguments import parse_args


def main():
    args = parse_args()

    # Set the GPU to use
    torch.cuda.set_device(args.gpu)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    vqa_loader = dataset.get_train_dataloader(osp.expanduser(args.annotations),
                                              osp.expanduser(args.questions),
                                              args.images, args, raw_images=args.raw_images,
                                              transforms=transform)
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

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.lr_decay)

    if args.visualize:
        vis = visualize.Visualizer(args.port)
    else:
        vis = None

    print("Beginning training")
    print("#"*80)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        trainer.train(model, vqa_loader, criterion, optimizer, epoch, args, vis=vis)
        trainer.evaluate(model, val_loader, criterion, epoch, args, vis=vis)

    print("Training complete!")


if __name__ == "__main__":
    main()
