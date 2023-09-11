import math
import os
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def validation(model: torch.nn.Module,
               data_loader: Iterable,
               device: torch.device, epoch: int,
               log_writer=None,
               visual=False,
               args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'validation:'
    print_freq = 10
    accum_iter = args.accum_iter
    sum_acc = None
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device)
        with torch.no_grad():
            loss, pred, acc = model(samples, labels)
        if sum_acc == None:
            sum_acc = acc
        else:
            sum_acc += acc
        loss_value = loss.item()
        deviation = float(sum_acc) / len(data_loader)
        metric_logger.update(loss=loss_value)
        if visual:
            if acc > 30:
                print(acc)
                print(pred)
                # os.system("pause")

    if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        """ We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        log_writer.add_scalar('val_deviation', deviation, epoch)
    print("Averaged deviation: ", deviation)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
