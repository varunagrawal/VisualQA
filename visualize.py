import visdom

class Visualizer:
    def __init__(self, port):
        self.visdom = visdom.Visdom(port)

    def update_loss(self, loss):
        """
        Update the loss graph on Visdom
        :param loss: Tensor containing the batch loss
        :return:
        """
        pass
