import os
import torch
import copy

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)

class WriterCol(object):

    def __init__(self, writers):
        self.writers = writers

    def add_scalar(self, name, val, step):
        for w in self.writers.values():
            w.add_scalar(name, val, step)

    def add_histogram(self, name, val, step):
        for w in self.writers.values():
            w.add_histogram(name, val, step)

def log_result(writer, name, res, step):
    writer.add_scalar("{}/loss".format(name),           res['loss'],           step)
    writer.add_scalar("{}/acc_perc".format(name),       res['accuracy'],       step)
    writer.add_scalar("{}/err_perc".format(name), 100. - res['accuracy'],      step)

def train_batch(epoch, batch_idx, loss_sum, correct, ttl,
                input, target, model, criterion, optimizer,
                w_q, g_q, acc_q, writer,
                quantize_momentum=True):
    batch_size = input.shape[0]

    input = input.cuda(async=True)
    target = target.cuda(async=True)
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    output = model(input_var)
    loss = criterion(output, target_var)

    optimizer.zero_grad()
    loss.backward()

    # gradient quantization
    if g_q != None:
        for name, p in model.named_parameters():
            p.grad.data = g_q(p.grad.data).data

    # use accumulator to add gradient
    for name, param in model.named_parameters():
        param.data = model.weight_acc[name]

    optimizer.step()

    # quantize accumulator and quantize weight
    for name, param in model.named_parameters():
        model.weight_acc[name] = acc_q(param.data).data
        param.data = w_q(model.weight_acc[name]).data

    if quantize_momentum and g_q != None:
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = optimizer.state[p]
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'] = g_q(param_state['momentum_buffer'])

    # Weight quantization
    if w_q != None:
        for name, p in model.named_parameters():
            p.data = w_q(p.data).data

    loss_sum += loss.cpu().item() * input.size(0)
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target_var.data.view_as(pred)).sum()
    ttl += input.size()[0]

    return loss_sum, correct, ttl

def train_epoch(loader, model, criterion, optimizer, weight_quantizer, grad_quantizer,
                writer, epoch, quant_bias=True, quant_bn=True,
                quantize_momentum=True):
    model.train()

    loss_sum = 0.0
    correct, correct_noise = 0.0, 0.0
    ttl = 0
    for i, (input, target) in enumerate(loader):
        model.train()
        loss_sum, correct, correct_noise, ttl = train_batch(
            epoch, i, loss_sum, correct, correct_noise, ttl,
            input, target, model, criterion, optimizer,
            weight_quantizer, grad_quantizer, writer,
            quant_bias=quant_bias, quant_bn=quant_bn,
            quantize_momentum=quantize_momentum, flatness_loss=flatness_loss,
            alpha=alpha, sigma2=sigma2, nsamples=nsamples,
            alternate_fl=alternate_fl)

    correct = correct.cpu().item()
    correct_noise = correct_noise.cpu().item()
    return {
        'loss': loss_sum / float(ttl),
        'accuracy': correct / float(ttl) * 100.0,
        'accuracy_noise': correct_noise / float(ttl) * 100.0,
    }


def eval(loader, model, criterion):
    loss_sum = 0.0
    correct = 0.0
    correct_noise = 0.0

    model.eval()
    cnt = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.cuda(async=True)
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var)

            loss_sum += loss.data.cpu().item() * input.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target_var.data.view_as(pred)).sum()

            cnt += int(input.size()[0])

    correct = correct.cpu().item()
    return {
        'loss': loss_sum / float(cnt),
        'accuracy': correct / float(cnt) * 100.0,
    }

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
