from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, Predictor
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset
from losses import BCE_softlabels, sigmoid_rampup, get_losses_unlabeled
import time


# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--method', type=str, default='CDAC', choices=['CDAC'], help='CDAC is proposed method')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations to train (default: 50000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--lr_f', type=float, default=1.0, metavar='LR_F', help='learning rate (default: 1.0)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication(default: 0.1)')
parser.add_argument('--T', type=float, default=0.05, metavar='T', help='temperature')
parser.add_argument('--save_check', action='store_true', default=True, help='save checkpoint or not')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='alexnet', help='which network to use')
parser.add_argument('--source', type=str, default='real', help='source domain')
parser.add_argument('--target', type=str, default='sketch', help='target domain')
parser.add_argument('--dataset', type=str, default='multi', choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--num', type=int, default=3, help='number of labeled examples in the target')
parser.add_argument('--rampup_length', type=int, default=20000,
                    help='ramp consistency loss weight up during first n training steps')
parser.add_argument('--rampup_coef', type=float, default=30.0, help='coefficient of consistency loss')
parser.add_argument('--topk', default=5, type=int, help='top-k indices of rank ordered feature elements')
parser.add_argument('--threshold', default=0.95, type=float, help='threshold of pseudo labeling')
parser.add_argument('--remark', type=str, default='', help='remark')
args = parser.parse_args()

torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' % (
args.dataset, args.source, args.target, args.num, args.net))
source_dataset, target_dataset, target_dataset_unl, target_dataset_val, target_dataset_test, class_list = return_dataset(
    args)


def folder_preparation(args):
    import datetime
    nowtime = datetime.datetime.now().strftime('%m%d%H%M%S')

    main_path = 'record/%s_%s_%s_net_%s_%s_to_%s_num_%s_%s' % (nowtime, args.dataset, args.method, args.net, args.source,
                              args.target, args.num, args.remark)
    if not os.path.exists(main_path):
        os.makedirs(main_path)

    # logs saving
    # logs_file = os.path.join(main_path,
                             # 'logs_%s_net_%s_%s_to_%s_num_%s' %
                             # (args.method, args.net, args.source,
                              # args.target, args.num))
    logs_file = os.path.join(main_path, 'logs')
                              
    # checkpath saving
    checkpath = os.path.join(main_path, 'checkpath')
    if not os.path.exists(checkpath):
        os.makedirs(checkpath)

    return main_path, logs_file, checkpath


main_path, logs_file, checkpath = folder_preparation(args)
print("Main path to save: {}".format(main_path))

if args.net == 'resnet34':
    G = resnet34()
    inc = 512
    bs = 24
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
    bs = 32
else:
    raise ValueError('Model cannot be recognized.')

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

F1 = Predictor(num_class=len(class_list), inc=inc, temp=args.T)
weights_init(F1)

lr = args.lr

G = nn.DataParallel(G)
F1 = nn.DataParallel(F1)
G = G.to(device)
F1 = F1.to(device)

im_data_s = torch.FloatTensor(1)
im_data_bar_s = torch.FloatTensor(1)
im_data_bar2_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_bar_t = torch.FloatTensor(1)
im_data_bar2_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
im_data_bar_tu = torch.FloatTensor(1)
im_data_bar2_tu = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)

im_data_s = im_data_s.to(device)
im_data_bar_s = im_data_bar_s.to(device)
im_data_bar2_s = im_data_bar2_s.to(device)
im_data_t = im_data_t.to(device)
im_data_bar_t = im_data_bar_t.to(device)
im_data_bar2_t = im_data_bar2_t.to(device)
im_data_tu = im_data_tu.to(device)
im_data_bar_tu = im_data_bar_tu.to(device)
im_data_bar2_tu = im_data_bar2_tu.to(device)
gt_labels_s = gt_labels_s.to(device)
gt_labels_t = gt_labels_t.to(device)

im_data_s = Variable(im_data_s)
im_data_bar_s = Variable(im_data_bar_s)
im_data_bar2_s = Variable(im_data_bar2_s)
im_data_t = Variable(im_data_t)
im_data_bar_t = Variable(im_data_bar_t)
im_data_bar2_t = Variable(im_data_bar2_t)
im_data_tu = Variable(im_data_tu)
im_data_bar_tu = Variable(im_data_bar_tu)
im_data_bar2_tu = Variable(im_data_bar2_tu)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)

opt = {}
opt["logs_file"] = logs_file
opt["checkpath"] = checkpath
opt["class_list"] = class_list

source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs, num_workers=3, shuffle=True, drop_last=True)
target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=min(bs, len(target_dataset)), num_workers=3,
                                            shuffle=True, drop_last=True)
target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl, batch_size=bs * 2, num_workers=3, shuffle=True,
                                                drop_last=True)
target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
                                                num_workers=3, shuffle=True, drop_last=True)
target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs * 2, num_workers=3, shuffle=True,
                                                 drop_last=True)


def train(device, opt):
    G.train()
    F1.train()

    optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=args.lr_f, momentum=0.9, weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])

    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)

    best_acc = 0

    BCE = BCE_softlabels().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    start_time = time.time()
    for step in range(all_step):

        rampup = sigmoid_rampup(step, args.rampup_length)
        w_cons = args.rampup_coef * rampup

        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)
        lr_f = optimizer_f.param_groups[0]['lr']
        lr_g = optimizer_g.param_groups[0]['lr']

        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)

        # load labeled source data
        x_s, target_s = data_s[0], data_s[1]
        im_data_s = x_s.to(device)
        gt_labels_s = target_s.to(device)

        # load labeled target data
        x_t, target_t = data_t[0], data_t[1]
        im_data_t = x_t.to(device)
        gt_labels_t = target_t.to(device)

        # load unlabeled target data
        x_tu, x_bar_tu, x_bar2_tu = data_t_unl[0], data_t_unl[3], data_t_unl[4]
        im_data_tu = x_tu.to(device)
        im_data_bar_tu = x_bar_tu.to(device)
        im_data_bar2_tu = x_bar2_tu.to(device)

        zero_grad_all()
        # construct losses for overall labeled data
        data = torch.cat((im_data_s, im_data_t), 0)
        target = torch.cat((gt_labels_s, gt_labels_t), 0)
        output = G(data)  # [batchsize, num_classes]
        out1 = F1(output)  # [batchsize, ]
        ce_loss = criterion(out1, target)

        ce_loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()

        # construct losses for unlabeled target data
        aac_loss, pl_loss, con_loss = get_losses_unlabeled(args, G, F1, im_data=im_data_tu, im_data_bar=im_data_bar_tu,
                                                           im_data_bar2=im_data_bar2_tu, target=None, BCE=BCE,
                                                           w_cons=w_cons, device=device)
        loss = aac_loss + pl_loss + con_loss

        loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()

        if step % args.log_interval == 0:
            log_train = 'S {} T {} Train Ep: {} lr_f{:.6f} lr_g{:.6f}\n'.format(args.source, args.target,
                                                                                step, lr_f, lr_g)
            print(log_train)
            with open(opt["logs_file"], 'a') as f:
                f.write(log_train)

        if (step % args.save_interval) == 0 and step > 0 or (step == all_step - 1):
            loss_test, acc_test = test(target_loader_test)
            loss_val, acc_val = test(target_loader_val)
            G.train()
            F1.train()
            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test

            cur_time = time.time() - start_time
            print('Current acc test %f best acc test %f best acc val %f time cost %f sec.' % (
            acc_test, best_acc_test, acc_val, cur_time))

            with open(opt["logs_file"], 'a') as f:
                f.write('step %d current %f best %f final %f time cost %f sec.\n\n' % (
                step, acc_test, best_acc_test, acc_val, cur_time))

            G.train()
            F1.train()
            if args.save_check:
                print('Saving model')
                torch.save(G.state_dict(), os.path.join(opt["checkpath"],
                                                        "G_iter_model_{}_to_{}_step_{}.pth.tar".format(args.source,
                                                                                                       args.target,
                                                                                                       step)))
                torch.save(F1.state_dict(), os.path.join(opt["checkpath"],
                                                         "F1_iter_model_{}_to_{}_step_{}.pth.tar".format(args.source,
                                                                                                         args.target,
                                                                                                         step)))
            start_time = time.time()


def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().to(device)
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t = data_t[0].to(device)
            gt_labels_t = data_t[1].to(device)
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} F1 ({:.0f}%)\n'.format(test_loss, correct, size,
                                                                                    100. * correct / size))
    return test_loss.data, 100. * float(correct) / size


train(device=device, opt=opt)
