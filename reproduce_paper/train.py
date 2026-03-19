import time
import argparse
import logging
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from scipy.stats import gmean
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboard_logger import Logger
from resnet1d_wang_fds import resnet1d_wang_fds
from loss import *
from datasets import PTBXLDataset
from utils import *

import os
os.environ["KMP_WARNINGS"] = "FALSE"


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# imbalanced related
# LDS
parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number')
parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')
# FDS
parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
parser.add_argument('--fds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
parser.add_argument('--fds_ks', type=int, default=9, help='FDS kernel size: should be odd number')
parser.add_argument('--fds_sigma', type=float, default=1, help='FDS gaussian/laplace kernel sigma')
parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket considered for FDS')
parser.add_argument('--bucket_start', type=int, default=3, choices=[0, 3],
                    help='minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB')
parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

# re-weighting: SQRT_INV / INV
parser.add_argument('--reweight', type=str, default='none', choices=['none', 'sqrt_inv', 'inverse'], help='cost-sensitive reweighting scheme')
# two-stage training: RRT
parser.add_argument('--retrain_fc', action='store_true', default=False, help='whether to retrain last regression layer (regressor)')

# training/optimization related
parser.add_argument('--dataset', type=str, default='1000_timesteps', choices=["5000_timesteps","1000_timesteps"], help='dataset name') # chnage 
parser.add_argument('--data_dir', type=str, default='./data', help='data directory') #change 
parser.add_argument('--model', type=str, default='resnet1d_wang_fds', choices=["resnet1d_wang_fds", "inception1d_fds"], help='model name')
parser.add_argument('--store_root', type=str, default='checkpoint_inception', help='root path for storing checkpoints, logs')
parser.add_argument('--store_name', type=str, default='', help='experiment store name')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--loss', type=str, default='l1', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'], help='training loss type')
# Ranking regularizer related
parser.add_argument('--ranking', action='store_true', default=False, help='whether to enable ranking regularizer')
parser.add_argument('--ranking_lambda', type=float, default=0.1, help='ranking regularizer lambda parameter')
parser.add_argument('--ranking_weight', type=float, default=0.1, help='weight for ranking regularizer loss')
# ConR related
parser.add_argument('--conr', action='store_true', default=False, help='whether to enable ConR (Contrastive Regression) loss')
parser.add_argument('--conr_w', type=float, default=1.0, help='ConR window parameter for positive pairs')
parser.add_argument('--conr_weight', type=float, default=0.1, help='weight for ConR loss term')
parser.add_argument('--conr_e', type=float, default=0.01, help='ConR exponential coefficient for label distance weighting')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--epoch', type=int, default=90, help='number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*', default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--print_freq', type=int, default=50, help='logging frequency')
# parser.add_argument('--ecg_downsampliing_factor', type=int, default=1, help='image size used in training')
parser.add_argument('--workers', type=int, default=20, help='number of workers used in data loading')
# checkpoints
parser.add_argument('--resume', type=str, default='', help='checkpoint file path to resume training')
parser.add_argument('--pretrained', type=str, default='', help='checkpoint file path to load backbone weights')
parser.add_argument('--evaluate', action='store_true', help='evaluate only flag')

parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()

args.start_epoch, args.best_loss = 0, 1e5


if len(args.store_name):
    args.store_name = f'_{args.store_name}'
if not args.lds and args.reweight != 'none':
    args.store_name += f'_{args.reweight}'
if args.lds:
    args.store_name += f'_lds_{args.lds_kernel[:3]}_{args.lds_ks}'
    if args.lds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.lds_sigma}'
if args.fds:
    args.store_name += f'_fds_{args.fds_kernel[:3]}_{args.fds_ks}'
    if args.fds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.fds_sigma}'
    args.store_name += f'_{args.start_update}_{args.start_smooth}_{args.fds_mmt}'
if args.retrain_fc:
    args.store_name += f'_retrain_fc'
if args.ranking:
    args.store_name += f'_ranking_{args.ranking_lambda}_{args.ranking_weight}'
if args.conr:
    args.store_name += f'_conr_{args.conr_w}_{args.conr_weight}_{args.conr_e}'
args.store_name = f"{args.dataset}_{args.model}{args.store_name}_{args.optimizer}_{args.loss}_{args.lr}_{args.batch_size}"

prepare_folders(args)

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.store_root, args.store_name, 'training.log')),
        logging.StreamHandler()
    ])
print = logging.info
print(f"Args: {args}")
print(f"Store name: {args.store_name}")

tb_logger = Logger(logdir=os.path.join(args.store_root, args.store_name), flush_secs=2)


def main():
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    # Data
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']

    train_dataset = PTBXLDataset(data_dir=args.data_dir, df=df_train, split='train',
                          reweight=args.reweight, lds=args.lds, lds_kernel=args.lds_kernel, lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)
    val_dataset = PTBXLDataset(data_dir=args.data_dir, df=df_val, split='val')
    test_dataset = PTBXLDataset(data_dir=args.data_dir, df=df_test, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")

    # Model
    print('=====> Building model...')
    model = resnet1d_wang_fds(input_channels=12, fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                     start_update=args.start_update, start_smooth=args.start_smooth,
                     kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt)
    model = torch.nn.DataParallel(model).cuda()

    # evaluate only
    if args.evaluate:
        assert args.resume, 'Specify a trained model using [args.resume]'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"===> Checkpoint '{args.resume}' loaded (epoch [{checkpoint['epoch']}]), testing...")
        validate(test_loader, model, train_labels=train_labels, prefix='Test')
        return

    if args.retrain_fc:
        assert args.reweight != 'none' and args.pretrained
        print('===> Retrain last regression layer only!')
        for name, param in model.named_parameters():
            if 'fc' not in name and 'linear' not in name:
                param.requires_grad = False

    # Loss and optimizer
    if not args.retrain_fc:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        # optimize only the last linear layer
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        names = list(filter(lambda k: k is not None, [k if v.requires_grad else None for k, v in model.module.named_parameters()]))
        assert 1 <= len(parameters) <= 2  # fc.weight, fc.bias
        print(f'===> Only optimize parameters: {names}')
        optimizer = torch.optim.Adam(parameters, lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'linear' not in k and 'fc' not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')
        print(f'===> Pre-trained model loaded: {args.pretrained}')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume) if args.gpu is None else \
                torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
            args.start_epoch = checkpoint['epoch']
            args.best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"===> Loaded checkpoint '{args.resume}' (Epoch [{checkpoint['epoch']}])")
        else:
            print(f"===> No checkpoint found at '{args.resume}'")

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch, args)
        train_loss = train(train_loader, model, optimizer, epoch)
        val_loss_mse, val_loss_l1, val_loss_gmean = validate(val_loader, model, train_labels=train_labels)

        loss_metric = val_loss_mse if args.loss == 'mse' else val_loss_l1
        is_best = loss_metric < args.best_loss
        args.best_loss = min(loss_metric, args.best_loss)
        print(f"Best {'L1' if 'l1' in args.loss else 'MSE'} Loss: {args.best_loss:.3f}")
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'model': args.model,
            'best_loss': args.best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)
        print(f"Epoch #{epoch}: Train loss [{train_loss:.4f}]; "
              f"Val loss: MSE [{val_loss_mse:.4f}], L1 [{val_loss_l1:.4f}], G-Mean [{val_loss_gmean:.4f}]")

        tb_logger.log_value('train_loss', train_loss, epoch)
        tb_logger.log_value('val_loss_mse', val_loss_mse, epoch)
        tb_logger.log_value('val_loss_l1', val_loss_l1, epoch)
        tb_logger.log_value('val_loss_gmean', val_loss_gmean, epoch)

    # test with best checkpoint
    print("=" * 120)
    print("Test best model on testset...")
    checkpoint = torch.load(f"{args.store_root}/{args.store_name}/ckpt.best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded best model, epoch {checkpoint['epoch']}, best val loss {checkpoint['best_loss']:.4f}")
    test_loss_mse, test_loss_l1, test_loss_gmean = validate(test_loader, model, train_labels=train_labels, prefix='Test')
    print(f"Test loss: MSE [{test_loss_mse:.4f}], L1 [{test_loss_l1:.4f}], G-Mean [{test_loss_gmean:.4f}]\nDone")


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.4f')
    losses = AverageMeter(f'Loss ({args.loss.upper()})', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()
    for idx, (inputs, targets, weights) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs, targets, weights = \
            inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True), weights.cuda(non_blocking=True)
        if args.fds:
            outputs, features = model(inputs, targets, epoch)
        else:
            outputs = model(inputs, targets, epoch)
            features = None

        loss = globals()[f"weighted_{args.loss}_loss"](outputs, targets, weights)
        
        # Add ranking regularizer if enabled
        if args.ranking and features is not None:
            ranking_loss = ranking_regularizer_loss(features, targets, args.ranking_lambda)
            loss += args.ranking_weight * ranking_loss
        
        # Add ConR loss if enabled
        if args.conr and features is not None:
            try:
                conr_loss = ConR(features, targets, outputs.squeeze(), w=args.conr_w, weights=weights, e=args.conr_e)
                if not (torch.isnan(conr_loss) or torch.isinf(conr_loss)):
                    loss += args.conr_weight * conr_loss
            except Exception as e:
                print(f"Warning: ConR loss computation failed: {e}. Skipping ConR for this batch.")
        
        assert not (np.isnan(loss.item()) or loss.item() > 1e6), f"Loss explosion: {loss.item()}"

        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if idx % args.print_freq == 0:
            progress.display(idx)

    if args.fds and epoch >= args.start_update:
        print(f"Create Epoch [{epoch}] features of all training data...")
        encodings, labels = [], []
        with torch.no_grad():
            for (inputs, targets, _) in tqdm(train_loader):
                inputs = inputs.cuda(non_blocking=True)
                outputs, feature = model(inputs, targets, epoch)
                encodings.extend(feature.data.squeeze().cpu().numpy())
                labels.extend(targets.data.squeeze().cpu().numpy())

        encodings, labels = torch.from_numpy(np.vstack(encodings)).cuda(), torch.from_numpy(np.hstack(labels)).cuda()
        model.module.FDS.update_last_epoch_stats(epoch)
        model.module.FDS.update_running_stats(encodings, labels, epoch)

    return losses.avg


def validate(val_loader, model, train_labels=None, prefix='Val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
    losses_l1 = AverageMeter('Loss (L1)', ':.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_mse, losses_l1],
        prefix=f'{prefix}: '
    )

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')

    model.eval()
    losses_all = []
    preds, labels = [], []
    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, _) in enumerate(val_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = model(inputs)

            preds.extend(outputs.data.cpu().numpy())
            labels.extend(targets.data.cpu().numpy())

            loss_mse = criterion_mse(outputs, targets)
            loss_l1 = criterion_l1(outputs, targets)
            loss_all = criterion_gmean(outputs, targets)
            losses_all.extend(loss_all.cpu().numpy())

            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_freq == 0:
                progress.display(idx)

        shot_dict = shot_metrics(np.hstack(preds), np.hstack(labels), train_labels)
        loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
        print(f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}")
        print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\t"
              f"L1 {shot_dict['many']['l1']:.3f}\tG-Mean {shot_dict['many']['gmean']:.3f}")
        print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\t"
              f"L1 {shot_dict['median']['l1']:.3f}\tG-Mean {shot_dict['median']['gmean']:.3f}")
        print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\t"
              f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.3f}")

    return losses_mse.avg, losses_l1.avg, loss_gmean




def shot_metrics(preds, labels, train_labels=None):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif not isinstance(preds, np.ndarray):
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')
    
    preds = preds.astype(float)
    labels = labels.astype(float)
    
    groups = {
        'many':   {'mse': [], 'l1': [], 'cnt': 0},   # age <= 30
        'median': {'mse': [], 'l1': [], 'cnt': 0},   # 30 < age < 60
        'low':    {'mse': [], 'l1': [], 'cnt': 0},   # age >= 60
    }
    
    for p, y in zip(preds, labels):
        err = p - y
        abs_err = abs(err)
        
        if y <= 30:
            g = 'many'
        elif y < 60:
            g = 'median'
        else:
            g = 'low'
        
        groups[g]['mse'].append(err ** 2)
        groups[g]['l1'].append(abs_err)
        groups[g]['cnt'] += 1
    
    shot_dict = {}
    for g in groups:
        if groups[g]['cnt'] > 0:  
            shot_dict[g] = {
                'mse': np.mean(groups[g]['mse']),  
                'l1': np.mean(groups[g]['l1']),
                'gmean': gmean(groups[g]['l1']).astype(float)
            }
        else:
            shot_dict[g] = {'mse': np.nan, 'l1': np.nan, 'gmean': np.nan}
    
    return shot_dict

if __name__ == '__main__':
    main()