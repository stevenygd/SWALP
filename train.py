import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import utils
import data
import tabulate
import models
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from collections import defaultdict

parser = argparse.ArgumentParser(description='SWALP training codes.')
parser.add_argument('--dir', type=str, default=None,
                    help='training directory (default: None)')
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='dataset name: CIFAR10 or CIFAR100', choices=['CIFAR10', 'CIFAR100'])
parser.add_argument('--data_path', type=str, default="./data", required=True, metavar='PATH',
                    help='path to datasets location (default: "./data")')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)',
                    choices=['VGG16LP', 'VGG16BNLP', 'VGG19LP', 'VGG19BNLP', 'PreResNet164LP', 'PreResNet20LP'])
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N',
                    help='save frequency (default: 25)')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--quantize-momentum', action="store_true", default=True,
                    help="use low precision momentum")
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--swa_start', type=float, default=161, metavar='N',
                    help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR',
                    help='SWA LR (default: 0.05)')
parser.add_argument('--seed', type=int, default=200, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log_name', type=str, default='', metavar='S',
                    help="Name for the log dir")
parser.add_argument('--quant-type', type=str, default='stochastic', metavar='S',
                    help='rounding method, stochastic or nearest ', choices=['stochastic', 'nearest'])
parser.add_argument('--small-block', type=str, default="None",
                    help='Small block configuration.', choices=["None", "Conv", "FC"])
parser.add_argument('--block-dim', type=str, default="B",
                    help='Block quantizing dimensions.', choices=["B", "BC"])
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

numbers = ["weight", "grad", "activate", "error", "acc"]
for num in numbers:
    parser.add_argument('--wl-{}'.format(num), type=int, default=-1, metavar='N',
                        help='word length in bits for {}; -1 if full precision.'.format(num))
args = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

name = os.path.join(args.log_name, "seed{}".format(args.seed))
dir_name = os.path.join("./checkpoint", name)

print('Checkpoint directory {}'.format(dir_name))
os.makedirs(dir_name, exist_ok=True)
with open(os.path.join(dir_name, 'command.sh'), 'w') as f:
    f.write('python '+' '.join(sys.argv))
    f.write('\n')
time_str = time.strftime("%m_%d_%H_%M")
log_name = "runs/{}_{}".format(name, time_str)

print('Tensorboard loggint at %s' % log_name)
os.makedirs(log_name, exist_ok=True)
writer = SummaryWriter(log_dir=log_name)

print("Prepare data loaders:")
num_classes = data.num_classes_dict[args.dataset]
loaders = data.get_data(args.dataset, args.data_path, args.batch_size, args.num_workers)

print("Prepare quantizers:")
print("Block rounding, W:{}, A:{}, G:{}, E:{}, Acc:{}".format(args.wl_weight,args.wl_activate,
                                                              args.wl_grad, args.wl_error, args.wl_acc))
print("lr init: {}".format(args.lr_init))
print("swa start: {} swa lr: {}".format(args.swa_start, args.swa_lr))

weight_quantizer = lambda x : models.quantize_block(
        x, args.wl_weight, -1, args.quant_type, args.small_block, args.block_dim)
acc_quantizer = lambda x : models.quantize_block(
        x, args.wl_acc, -1, args.quant_type, args.small_block, args.block_dim)
grad_quantizer = lambda x : models.quantize_block(
        x, args.wl_grad, -1, args.quant_type, args.small_block, args.block_dim)


# Build model
print('Model: {}'.format(args.model))
model_cfg = getattr(models, args.model)
quant = lambda : models.BlockQuantizer(args.wl_activate, args.wl_error, args.quant_type,
                                       args.small_block, args.block_dim)
model_cfg.kwargs.update({"quant":quant, "writer":writer})
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()

# append weight accumulator
model.weight_acc = {}
for name, param in model.named_parameters():
    model.weight_acc[name] = param.data.clone().cuda()

print('Prepare SWA training')
swa_n = 0
quant = lambda : models.BlockQuantizer(-1, -1, args.quant_type,
                                       args.small_block, args.block_dim)
model_cfg.kwargs.update({"quant":quant, "writer":writer})
swa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
swa_model.cuda()

# Learning rate schedule
def schedule(epoch):
    t = (epoch) / args.swa_start
    lr_ratio = 0.01
    lr_swa_ratio = args.swa_lr / args.lr_init
    if t <= 0.5:
       factor = 1.0
    elif t <= 0.9:
       factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    elif t < 1.0:
       factor = lr_ratio
    else:
       factor = lr_swa_ratio
    return args.lr_init * factor

# Loss and optimizer
criterion = F.cross_entropy
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd,
)

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']-1
    resume_keys = list(checkpoint['state_dict'])
    model_keys = list(model.state_dict())
    matched_state_dict = {
        model_keys[i]:checkpoint['state_dict'][k] for i,k in enumerate(resume_keys)}
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(matched_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for name, param in model.named_parameters():
        model.weight_acc[name] = param.data.clone().cuda()


# Prepare logging
columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']
columns = columns[:-1] + ['swa_te_acc'] + columns[-1:]
swa_res = {'loss': None, 'accuracy': None}

sgd_acc = 0
swa_acc = 0
for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()
    lr = schedule(epoch)
    writer.add_scalar("lr", lr, epoch)
    utils.adjust_learning_rate(optimizer, lr)

    model.train()
    loss_sum = 0.0
    correct = 0.0
    ttl = 0
    for i, (input, target) in enumerate(loaders['train']):
        model.train()
        loss_sum, correct, ttl = utils.train_batch(
            epoch, i, loss_sum, correct, ttl,
            input, target, model, criterion, optimizer,
            weight_quantizer, grad_quantizer, acc_quantizer, writer,
            quantize_momentum=args.quantize_momentum)

    if (epoch + 1) >= args.swa_start:
        utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
        swa_n += 1

    correct = correct.cpu().item()
    train_res = {
        'loss': loss_sum / float(ttl),
        'accuracy': correct / float(ttl) * 100.0,
    }
    utils.log_result(writer, "train", train_res, epoch+1)

    # Validation : SGD performance
    test_res = defaultdict(lambda : None)
    utils.bn_update(loaders['train'], model)
    test_res.update(utils.eval(loaders['test'], model, criterion))
    utils.log_result(writer, "test", test_res, epoch+1)
    if (epoch + 1) == args.swa_start:
        sgd_acc = test_res['accuracy']

    # Validation : SWA performance
    swa_te_res = defaultdict(lambda : None)
    if (epoch + 1) >= args.swa_start:
        utils.bn_update(loaders['train'], swa_model)
        swa_te_res.update(utils.eval(loaders['test'], swa_model, criterion))
        utils.log_result(writer, "test_swa", swa_te_res, epoch+1)
    if (epoch + 1) == args.epochs - 1:
        swa_acc = swa_te_res['accuracy']

    # Save Checkpoint
    if (epoch+1) % args.save_freq == 0 or (epoch+1) == args.swa_start:
        utils.save_checkpoint(
            dir_name,
            epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )

    # Log results to the terminal
    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'],
              test_res['loss'], test_res['accuracy'], time_ep]

    values = values[:-1] + [swa_te_res['accuracy']] + values[-1:]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)


# Save the final model
if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        dir_name,
        args.epochs,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )

swa_dir = os.path.join(dir_name, "swa")
os.makedirs(swa_dir, exist_ok=True)
utils.save_checkpoint(
    swa_dir,
    args.epochs,
    state_dict=swa_model.state_dict()
)

print("Done")
print("SGD Error : %s"%(100 - sgd_acc))
print("SWA Error : %s"%(100 - swa_acc))

