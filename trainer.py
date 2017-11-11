import torch
from torch.autograd import Variable


def train(model, dataloader, criterion, optimizer, epoch, args):
    if torch.cuda.is_available():
        model.cuda()

    model.train()

    # TODO use average meter for loss
    # TODO print average accuracy
    losses = []

    for idx, sample in enumerate(dataloader):
        q = sample['question']
        img = sample["visual"]
        ans_label = sample['answer_id']

        q = Variable(q).cuda()
        img = Variable(img).cuda()
        ans = Variable(ans_label).cuda()

        output = model(img, q)

        loss = criterion(output, ans)
        losses.append(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx > 0 and idx % args.print_freq == 0:
            print_state(idx, epoch, len(dataloader.dataset), losses=losses)

    save_checkpoint(model, args, epoch)


def evaluate(model, dataloader, criterion, epoch, args):
    # switch to evaluate mode
    model.eval()

    losses = []

    for i, sample in enumerate(dataloader):
        q = sample['question']
        img = sample["visual"]
        ans_label = sample['answer_id']

        q = Variable(q).cuda()
        img = Variable(img).cuda()
        ans = Variable(ans_label).cuda()

        output = model(img, q)

        loss = criterion(output, ans)

        losses.append(loss.data[0])

        if i > 0 and i % args.print_freq == 0:
            print_state(i, epoch, len(dataloader.datase), losses=losses)


def save_checkpoint(model, args, epoch):
    state = {
        "model": model,
        "args": args
    }
    filename = 'vqa_checkpoint_{0}.pth.tar'.format(epoch)
    torch.save(state, filename)


def print_state(idx, epoch, size, losses):
    if epoch >= 0:
        message = "Epoch: [{0}][{1}/{2}]\t".format(epoch, idx, size)
    else:
        message = "Test: [{0}/{1}]\t".format(idx, size)

    avg_loss = sum(losses) / len(losses)
    avg_acc = "Not Implemented"

    print(message +
          'Avg Loss {loss} \t'
          'Accuracy {acc}'.format(loss=avg_loss, acc=avg_acc))
