import time
import os
import shutil
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import torch


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch, writer, para):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        target = target.to(para['device'])
        input = input.to(para['device'])

        # compute output
        output = model(input)
        loss, loss1, loss2 = criterion(output, target, model)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        losses1.update(loss1.item(), input.size(0))
        losses2.update(loss2.item(), input.size(0))
        top1.update(100 - prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # input()
        if i % para['print_freq'] == 0:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

    # log to TensorBoard
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/loss1', losses1.avg, epoch)
    writer.add_scalar('train/loss2', losses2.avg, epoch)
    writer.add_scalar('train/err', top1.avg, epoch)

    return top1.avg


def validate(log_name, val_loader, model, criterion, epoch, writer, para):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        target = target.to(para['device'])
        input = input.to(para['device'])

        # compute output
        output = model.forward_test(input)
        loss, loss1, loss2 = criterion(output, target, model)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        losses1.update(loss1.item(), input.size(0))
        losses2.update(loss2.item(), input.size(0))
        top1.update(100 - prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % para['print_freq'] == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Err@1 {top1.avg:.3f}'.format(top1=top1))

    zero_num = model.num_0()
    zero_perc = zero_num / para['group_num']

    # log to TensorBoard
    writer.add_scalar('val_{}/num_0'.format(log_name), zero_num, epoch)
    writer.add_scalar('val_{}/perc_0'.format(log_name), zero_perc, epoch)
    writer.add_scalar('val_{}/loss'.format(log_name), losses.avg, epoch)
    writer.add_scalar('val_{}/loss1'.format(log_name), losses1.avg, epoch)
    writer.add_scalar('val_{}/loss2'.format(log_name), losses2.avg, epoch)
    writer.add_scalar('val_{}/err'.format(log_name), top1.avg, epoch)

    return top1.avg


def log_groups(model, writer, epoch):
    loga = model.loga.cpu().detach()
    z = model.l0_test(loga)
    index_nonzero = torch.where(z != 0)
    writer.add_text('selected_groups', str(index_nonzero), epoch)


def train_finetune(train_loader, model, criterion, optimizer, epoch, writer, para):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        target = target.to(para['device'])
        input = input.to(para['device'])

        # compute output
        output = model(input)
        loss = criterion(output, target, model)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(100 - prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # input()
        if i % para['print_freq'] == 0:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

    # log to TensorBoard
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/err', top1.avg, epoch)

    return top1.avg


def validate_finetune(log_name, val_loader, model, criterion, epoch, writer, para):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        target = target.to(para['device'])
        input = input.to(para['device'])

        # compute output
        output = model(input)
        loss = criterion(output, target, model)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(100 - prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % para['print_freq'] == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Err@1 {top1.avg:.3f}'.format(top1=top1))

    # log to TensorBoard
    writer.add_scalar('val_{}/loss'.format(log_name), losses.avg, epoch)
    writer.add_scalar('val_{}/err'.format(log_name), top1.avg, epoch)
    return top1.avg


def para2dir(para):
    path = ''
    for key in para:
        path += key
        path += '/'
        path += str(para[key])
        path += '/'
    path = path[:-1]
    return path


def get_log_writer(para):
    base_dir = para2dir(para)

    directory = 'logs/{}'.format(base_dir)
    if os.path.exists(directory):
        shutil.rmtree(directory)
        os.makedirs(directory)
    else:
        os.makedirs(directory)

    directory2 = 'masks/' + base_dir
    if not os.path.exists(directory2):
        os.makedirs(directory2)

    return SummaryWriter(directory), base_dir
