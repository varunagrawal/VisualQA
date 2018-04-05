import torch
import visdom


class Visualizer:
    def __init__(self, port):
        self.vis = visdom.Visdom(port=port)

        self.send(torch.zeros((1,)), torch.zeros((1,)), win="loss")
        self.send(torch.zeros((1,)), torch.zeros((1,)), win="val_loss")

    def send(self, x, y, win, update=False):
        try:
            if update:
                self.vis.line(X=x, Y=y, win=win, update="append")
            else:
                self.vis.line(X=x, Y=y, win=win)
        except (Exception,):
            # quietly settle down
            pass

    def update_loss(self, loss, epoch, iteration, data_size, win):
        """
        Update the loss graph on Visdom
        :param loss: Tensor containing the batch loss
        :param epoch:
        :param iteration:
        :param data_size:
        :param win:
        :return:
        """
        try:
            self.send(torch.ones((1,)) * (epoch * data_size + iteration),
                      loss.data.cpu(),
                      win, update=True)
        except (Exception,):
            # quietly settle down
            pass
