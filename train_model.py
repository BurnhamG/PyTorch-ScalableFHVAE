import argparse
import logging
import os


# alpha/discriminative weight of 10 was found to produce best results
def loss_function(lower_bound, log_qy, alpha=10.0):
    """
    Discriminative segment variational lower bound
    Segment variational lower bound plus the (weighted) discriminative objective
    """

    return -1 * torch.mean(lower_bound + alpha * log_qy)
