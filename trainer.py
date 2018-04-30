import torch
from metrics import accuracy
import os, os.path as osp


def train(model, dataloader, criterion, optimizer, epoch, args, vis=None):
    # Set the model to train mode
    model.train()
    # enable autograd tracking
    torch.set_grad_enabled(True)

    avg_loss = AverageMeter()
    avg_acc = AverageMeter()

    for idx, sample in enumerate(dataloader):
        q = sample['question']
        img = sample["image"]
        ans_label = sample['answer_id']

        q = q.cuda()
        img = img.cuda()
        ans = ans_label.cuda()

        output = model(img, q)

        loss = criterion(output, ans)
        avg_loss.update(loss.item(), q.size(0))

        acc = accuracy(output, ans)
        avg_acc.update(acc.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if vis and idx % args.visualize_freq == 0:
            vis.update_loss(loss, epoch, idx, len(dataloader), "loss")

        if idx > 0 and idx % args.print_freq == 0:
            print_state(idx, epoch, len(dataloader), avg_loss.avg, avg_acc.avg)

    save_checkpoint(model, args, epoch)


def evaluate(model, dataloader, criterion, epoch, args, vis=None):
    # switch to evaluate mode
    model.eval()
    # disable autograd tracking
    torch.set_grad_enabled(False)

    avg_loss = AverageMeter()
    avg_acc = AverageMeter()

    for i, sample in enumerate(dataloader):
        q = sample['question']
        img = sample["image"]
        ans_label = sample['answer_id']

        q = q.cuda()
        img = img.cuda()
        ans = ans_label.cuda()

        output = model(img, q)

        loss = criterion(output, ans)

        avg_loss.update(loss.item(), q.size(0))

        acc = accuracy(output, ans)
        avg_acc.update(acc.item())

        if vis and i % args.visualize_freq == 0:
            vis.update_loss(loss, epoch, i, len(dataloader), "val_loss")

        if i > 0 and i % args.print_freq == 0:
            print_state(i, -1, len(dataloader), avg_loss.avg, avg_acc.avg)


def save_checkpoint(model, args, epoch):
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    state = {
        "model": model.state_dict(),
        "args": args
    }
    filename = 'vqa_checkpoint_{0}_{1}.pth'.format(args.arch, epoch)
    torch.save(state, osp.join(args.save_dir, filename))


def print_state(idx, epoch, size, loss, acc):
    if epoch >= 0:
        message = "Epoch: [{0}][{1}/{2}]\t\t".format(epoch, idx, size)
    else:
        message = "Test: [{0}/{1}]\t\t".format(idx, size)

    print(message +
          'Loss {loss:.4f} \t'
          'Accuracy {acc:.4f}'.format(loss=loss, acc=acc))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
