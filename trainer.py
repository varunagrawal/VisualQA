import torch
from torch.autograd import Variable
from metrics import accuracy
import os, os.path as osp


def train(model, dataloader, criterion, optimizer, epoch, args, vis=None):
    # Set the model to train mode
    model.train()

    avg_loss = AverageMeter()
    avg_acc = AverageMeter()

    for idx, sample in enumerate(dataloader):
        q = sample['question']
        img = sample["visual"]
        ans_label = sample['answer_id']

        q = Variable(q).cuda()
        img = Variable(img).cuda()
        ans = Variable(ans_label).cuda()

        output = model(img, q)

        loss = criterion(output, ans)
        avg_loss.update(loss.data[0], q.size(0))

        acc = accuracy(output, ans)
        avg_acc.update(acc.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if vis:
            vis.update_loss(loss, epoch, idx, len(dataloader), "loss")

        if idx > 0 and idx % args.print_freq == 0:
            print_state(idx, epoch, len(dataloader), avg_loss, avg_acc)

    save_checkpoint(model, args, epoch)


def evaluate(model, dataloader, criterion, epoch, args, vis=None):
    # switch to evaluate mode
    model.eval()

    avg_loss = AverageMeter()
    avg_acc = AverageMeter()

    for i, sample in enumerate(dataloader):
        q = sample['question']
        img = sample["visual"]
        ans_label = sample['answer_id']

        q = Variable(q).cuda()
        img = Variable(img).cuda()
        ans = Variable(ans_label).cuda()

        output = model(img, q)

        loss = criterion(output, ans)

        acc = accuracy(output, ans)
        avg_acc.update(acc.data[0])

        avg_loss.update(loss.data[0], q.size(0))

        if vis:
            vis.update_loss(loss, epoch, i, len(dataloader), "val_loss")

        if i > 0 and i % args.print_freq == 0:
            print_state(i, epoch, len(dataloader), avg_loss, avg_acc)


def save_checkpoint(model, args, epoch):
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    state = {
        "model": model,
        "args": args
    }
    filename = 'vqa_checkpoint_{0}.pth.tar'.format(epoch)
    torch.save(state, osp.join(args.save_dir, filename))


def print_state(idx, epoch, size, avg_loss, avg_acc):
    if epoch >= 0:
        message = "Epoch: [{0}][{1}/{2}]\t\t".format(epoch, idx, size)
    else:
        message = "Test: [{0}/{1}]\t\t".format(idx, size)

    print(message +
          'Loss {loss:.4f} \t'
          'Accuracy {acc:.4f}'.format(loss=avg_loss.avg, acc=avg_acc.avg))


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
