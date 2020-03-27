# Logger classes originally from
# https://github.com/SeanNaren/deepspeech.pytorch/blob/master/logger.py

import os

import torch


def to_np(x):
    return x.cpu().numpy()


class VisdomLogger(object):
    def __init__(self, id, num_epochs):
        from visdom import Visdom

        self.viz = Visdom()
        self.opts = dict(
            title=id,
            ylabel="",
            xlabel="Epoch",
            legend=["Loss", "Lower Bound", "Discriminative Loss"],
        )
        self.viz_window = None
        self.epochs = torch.arange(1, num_epochs + 1)
        self.visdom_plotter = True

    def update(self, epoch, values):
        x_axis = self.epochs[0 : epoch + 1]
        y_axis = torch.stack(
            (
                values["loss_results"][: epoch + 1],
                values["lower_bound_results"][: epoch + 1],
                values["discrim_loss_results"][: epoch + 1],
            ),
            dim=1,
        )
        self.viz_window = self.viz.line(
            X=x_axis,
            Y=y_axis,
            opts=self.opts,
            win=self.viz_window,
            update="replace" if self.viz_window else None,
        )

    def load_previous_values(self, start_epoch, package):
        # Add all values except the iteration we're starting from
        self.update(start_epoch - 1, package)


class TensorBoardLogger(object):
    def __init__(self, id, log_dir, log_params):
        os.makedirs(log_dir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter

        self.id = id
        self.tensorboard_writer = SummaryWriter(log_dir)
        self.log_params = log_params

    def update(self, epoch, values, parameters=None):
        loss, lower_bound, discrim_loss = (
            values["loss_results"][epoch + 1],
            values["lower_bound_results"][epoch + 1],
            values["discrim_loss_results"][epoch + 1],
        )
        values = {
            "Avg Train Loss": loss,
            "Avg Lower Bound": lower_bound,
            "Avg Discriminative Loss": discrim_loss,
        }
        self.tensorboard_writer.add_scalars(self.id, values, epoch + 1)
        if self.log_params:
            for tag, value in parameters():
                tag = tag.replace(".", "/")
                self.tensorboard_writer.add_histogram(tag, to_np(value), epoch + 1)
                self.tensorboard_writer.add_histogram(
                    tag + "/grad", to_np(value.grad), epoch + 1
                )

    def load_previous_values(self, start_epoch, values):
        loss_results = values["loss_results"][:start_epoch]
        lb_results = values["lower_bound_results"][:start_epoch]
        dl_results = values["discrim_loss_results"][:start_epoch]

        for i in range(start_epoch):
            values = {
                "Avg Train Loss": loss_results[i],
                "Avg Lower Bound": lb_results[i],
                "Avg Discriminative Loss": dl_results[i],
            }
            self.tensorboard_writer.add_scalars(self.id, values, i + 1)
