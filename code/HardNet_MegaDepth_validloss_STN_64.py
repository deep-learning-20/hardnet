#!/usr/bin/python2 -utt
#-*- coding: utf-8 -*-
"""
This is HardNet local patch descriptor. The training code is based on PyTorch TFeat implementation
https://github.com/edgarriba/examples/tree/master/triplet
by Edgar Riba.
If you use this code, please cite 
@article{HardNet2017,
 author = {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title = "{Working hard to know your neighbor's margins:Local descriptor learning loss}",
     year = 2017}
(c) 2017 by Anastasiia Mishchuk, Dmytro Mishkin 
"""

from __future__ import division, print_function
import sys
from copy import deepcopy
import math
import argparse
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import copy
import PIL
from EvalMetrics import ErrorRateAt95Recall
from Losses import loss_HardNet, loss_random_sampling, loss_L2Net, global_orthogonal_regularization,loss_transform
from W1BS import w1bs_extract_descs_and_save
from Utils import L2Norm, cv2_scale, np_reshape
from Utils import str2bool
import torch.nn as nn
import torch.nn.functional as F
from lib.dataset_sanity import MegaDepthDataset
import sys
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

print(sys.path)

class CorrelationPenaltyLoss(nn.Module):
    def __init__(self):
        super(CorrelationPenaltyLoss, self).__init__()

    def forward(self, input):
        mean1 = torch.mean(input, dim=0)
        zeroed = input - mean1.expand_as(input)
        cor_mat = torch.bmm(torch.t(zeroed).unsqueeze(0), zeroed.unsqueeze(0)).squeeze(0)
        d = torch.diag(torch.diag(cor_mat))
        no_diag = cor_mat - d
        d_sq = no_diag * no_diag
        return torch.sqrt(d_sq.sum())/input.size(0)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch HardNet')
# Model options

parser.add_argument('--w1bsroot', type=str,
                    default='data/sets/wxbs-descriptors-benchmark/code/',
                    help='path to dataset')
parser.add_argument('--dataroot', type=str,
                    default='data/sets/',
                    help='path to dataset')
parser.add_argument('--enable-logging',type=str2bool, default=True,
                    help='output to tensorlogger')
parser.add_argument('--log-dir', default='data/logs/',
                    help='folder to output log')
parser.add_argument('--model-dir', default='data/models/',
                    help='folder to output model checkpoints')
parser.add_argument('--experiment-name', default= 'liberty_train/',
                    help='experiment path')
parser.add_argument('--training-set', default= 'liberty',
                    help='Other options: notredame, yosemite')
parser.add_argument('--loss', default= 'triplet_margin',
                    help='Other options: softmax, contrastive')
parser.add_argument('--batch-reduce', default= 'min',
                    help='Other options: average, random, random_global, L2Net')
parser.add_argument('--num-workers', default= 0, type=int,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory',type=bool, default= True,
                    help='')
parser.add_argument('--decor',type=str2bool, default = False,
                    help='L2Net decorrelation penalty')
parser.add_argument('--anchorave', type=str2bool, default=False,
                    help='anchorave')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--mean-image', type=float, default=0.443728476019,
                    help='mean of train dataset for normalization')
parser.add_argument('--std-image', type=float, default=0.20197947209,
                    help='std of train dataset for normalization')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=200, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--anchorswap', type=str2bool, default=True,
                    help='turns on anchor swap')
parser.add_argument('--batch-size', type=int, default=1024, metavar='BS',
                    help='input batch size for training (default: 1024)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='BST',
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--n-triplets', type=int, default=5000000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--gor',type=str2bool, default=False,
                    help='use gor')
parser.add_argument('--freq', type=float, default=10.0,
                    help='frequency for cyclic learning rate')
parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                    help='gor parameter')
parser.add_argument('--lr', type=float, default=10.0, metavar='LR',
                    help='learning rate (default: 10.0. Yes, ten is not typo)')
parser.add_argument('--fliprot', type=str2bool, default=True,
                    help='turns on flip and 90deg rotation augmentation')
parser.add_argument('--augmentation', type=str2bool, default=False,
                    help='turns on shift and small scale rotation augmentation')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: SGD)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

suffix = '{}_{}_{}'.format(args.experiment_name, args.training_set, args.batch_reduce)

if args.gor:
    suffix = suffix+'_gor_alpha{:1.1f}'.format(args.alpha)
if args.anchorswap:
    suffix = suffix + '_as'
if args.anchorave:
    suffix = suffix + '_av'
if args.fliprot:
        suffix = suffix + '_fliprot'

triplet_flag = (args.batch_reduce == 'random_global') or args.gor

dataset_names = ['liberty', 'notredame', 'yosemite']

TEST_ON_W1BS = False
# check if path to w1bs dataset testing module exists
if os.path.isdir(args.w1bsroot):
    sys.path.insert(0, args.w1bsroot)
    import utils.w1bs as w1bs
    TEST_ON_W1BS = True

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()

print (("NOT " if not args.cuda else "") + "Using cuda")

if args.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

# create loggin directory
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# set random seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

class HardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
        )
        #self.features.apply(weights_init)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
#29
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.MaxPool2d(2, stride=2),

            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.ReLU(True))
        #self.features.apply(weights_init)

        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2)
        )

        
    
        self.features.apply(weights_init)
        self.localization.apply(weights_init)
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        return
    
    def stn(self, x):
        xs = self.localization(x)
        #print("what's XS", xs.size())
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        print("learned theta value",theta[:5])
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x,theta

    #def get_transform()
    
    def input_norm(self,x):
        flat = x.view(x.size(0),x.size(1), -1)
        #print("what is this size",x.size(0))
        #print("what is this size",x.size())
        #print("what is this size",flat.size())

        mp = torch.mean(flat, dim=2)
        sp = torch.std(flat, dim=2) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).expand_as(x)
    def forward(self, input):
        """
        x=self.input_norm(input)
        print("what's Xsize here lol",x.size())

        x = self.stn(input)
        return x
        """
        
        x=self.input_norm(input)

        x,theta = self.stn(x)
        #print("FC weight",self.fc_loc[2].weight)
        #print("FC bias",self.fc_loc[2].weight)

        
        x_features = self.features(x)
        #print(x_features)
        x = x_features.view(x_features.size(0), -1)
        #x=input

        return L2Norm()(x),theta
        
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return


def train(train_loader, model, optimizer, epoch, logger, load_triplets  = False):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar:
    
        #print(np.array(data).shape)
        #print("data",data)
        print("STN weight", model.fc_loc[2].weight)
        print("STN weight_bias", model.fc_loc[2].bias)
        print("grad value",list(model.parameters())[0].grad)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print (name, param.data)
        if load_triplets:
            data_a, data_p, data_n = data
        else:
            data_a, data_p = data

        if args.cuda:
            data_a, data_p  = data_a.cuda(), data_p.cuda()
            data_a, data_p = Variable(data_a), Variable(data_p)
            out_a,theta_a = model(data_a)
            out_p,theta_p = model(data_p)
        if load_triplets:
            data_n  = data_n.cuda()
            data_n = Variable(data_n)
            out_n = model(data_n)

        if args.batch_reduce == 'L2Net':
            loss = loss_L2Net(out_a, out_p, anchor_swap = args.anchorswap,
                    margin = args.margin, loss_type = args.loss)
        elif args.batch_reduce == 'random_global':
            loss = loss_random_sampling(out_a, out_p, out_n,
                margin=args.margin,
                anchor_swap=args.anchorswap,
                loss_type = args.loss)
        else:
            loss = loss_HardNet(out_a, out_p,
                            margin=args.margin,
                            anchor_swap=args.anchorswap,
                            anchor_ave=args.anchorave,
                            batch_reduce = args.batch_reduce,
                            loss_type = args.loss)

        if args.decor:
            loss += CorrelationPenaltyLoss()(out_a)
            
        if args.gor:
            loss += args.alpha*global_orthogonal_regularization(out_a, out_n)
        
        
        loss1=loss_transform(theta_a,theta_p)
        loss2=loss+loss1
        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()
        adjust_learning_rate(optimizer)
        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.item()))

    if (args.enable_logging):
        logger.log_value('loss', loss.item()).step()

    try:
        os.stat('{}{}'.format(args.model_dir,suffix))
    except:
        os.makedirs('{}{}'.format(args.model_dir,suffix))

    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}{}/checkpoint_{}.pth'.format(args.model_dir,suffix,epoch))

def test(test_loader, model, epoch, logger, logger_test_name):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    print("pbar",pbar)
    for batch_idx, data in pbar:
    

        #print("testset",np.array(data).shape)
        #print("data",data)
        print("testset batch_id",batch_idx)
        print("test set size",len(test_loader.dataset))



        #print(data.size())
        #print("data",data)
        data_a, data_p = data   
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        

        label=torch.ones(args.batch_size)
        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)
        out_a,theta_a = model(data_a)
        out_p,theta_p = model(data_p)
        """        
        data_a_transformed=model.stn(data_a).cpu().detach().numpy()
        data_p_transformed=model.stn(data_p).cpu().detach().numpy()
        data_a=data_a.cpu().numpy()
        data_p=data_p.cpu().numpy()
        import pickle

        
        writefile = open("/cluster/scratch/tianhu/test/test_STN.pkl", "wb")
        pickle.dump([data_a,data_p,data_a_transformed,data_p_transformed], writefile)
        writefile.close()
        """
        
        if args.batch_reduce == 'L2Net':
            loss = loss_L2Net(out_a, out_p, anchor_swap = args.anchorswap,
                    margin = args.margin, loss_type = args.loss)
        elif args.batch_reduce == 'random_global':
            loss = loss_random_sampling(out_a, out_p, out_n,
                margin=args.margin,
                anchor_swap=args.anchorswap,
                loss_type = args.loss)
        else:
            loss = loss_HardNet(out_a, out_p,
                            margin=args.margin,
                            anchor_swap=args.anchorswap,
                            anchor_ave=args.anchorave,
                            batch_reduce = args.batch_reduce,
                            loss_type = args.loss)

        #dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        #distances.append(dists.data.cpu().numpy().reshape(-1,1))
        #ll = label.data.cpu().numpy().reshape(-1, 1)
        #labels.append(ll)
        #distances_1=np.array(distances)
        
        #print(distances_1.shape)
        

        #distances_1 = [item for sublist in distances for item in sublist]
        

        #distances_1=np.mean(np.array(distances_1))
       
        if batch_idx % args.log_interval == 0:
            print("heyheyhey b size", batch_idx)
            print("heyheyhey dataset size", len(test_loader))
            pbar.set_description(
                'Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(test_loader.dataset),
                           100. * batch_idx / len(test_loader),
                    loss))
    """

    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)


    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
    """
    #distances_1 = [item for sublist in distances for item in sublist]

    #distances=np.mean(np.array(distances_1))

    print('\33[91mTest set: mean loss: {:.8f}\n\33[0m'.format(loss))

    if (args.enable_logging):
        logger.log_value('test_meandistance', loss)
    return

def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = args.lr * (
        1.0 - float(group['step']) * float(args.batch_size) / (args.n_triplets * float(args.epochs)))
    return

def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer


def main(train_loader, test_loader, training_dataset, validation_dataset,model, logger, file_logger):
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))

    #if (args.enable_logging):
    #    file_logger.log_string('logs.txt', '\nparsed options:\n{}\n'.format(vars(args)))

    if args.cuda:
        model.cuda()

    optimizer1 = create_optimizer(model, args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))
            


    start = args.start_epoch
    end = start + args.epochs
    validation_dataset.build_dataset()
    print(len(validation_dataset))
    for epoch in range(start, end):
        training_dataset.build_dataset()

        # iterate over test loaders and test results
        train(train_loader, model, optimizer1, epoch, logger, triplet_flag)
            #validation_dataset.build_dataset()
        print("start validation")
        test(test_loader,model, epoch, logger, test_loader)
        
        if TEST_ON_W1BS :
            # print(weights_path)
            patch_images = w1bs.get_list_of_patch_images(
                DATASET_DIR=args.w1bsroot.replace('/code', '/data/W1BS'))
            desc_name = 'curr_desc'# + str(random.randint(0,100))
            
            DESCS_DIR = LOG_DIR + '/temp_descs/' #args.w1bsroot.replace('/code', "/data/out_descriptors")
            OUT_DIR = DESCS_DIR.replace('/temp_descs/', "/out_graphs/")

            for img_fname in patch_images:
                w1bs_extract_descs_and_save(img_fname, model, desc_name, cuda = args.cuda,
                                            mean_img=args.mean_image,
                                            std_img=args.std_image, out_dir = DESCS_DIR)


            force_rewrite_list = [desc_name]
            w1bs.match_descriptors_and_save_results(DESC_DIR=DESCS_DIR, do_rewrite=True,
                                                    dist_dict={},
                                                    force_rewrite_list=force_rewrite_list)
            if(args.enable_logging):
                w1bs.draw_and_save_plots_with_loggers(DESC_DIR=DESCS_DIR, OUT_DIR=OUT_DIR,
                                         methods=["SNN_ratio"],
                                         descs_to_draw=[desc_name],
                                         logger=file_logger,
                                         tensor_logger = logger)
            else:
                w1bs.draw_and_save_plots(DESC_DIR=DESCS_DIR, OUT_DIR=OUT_DIR,
                                         methods=["SNN_ratio"],
                                         descs_to_draw=[desc_name])
        #randomize train loader batches
        #train_loader, test_loaders2 = create_loaders(load_random_triplets=triplet_flag)


if __name__ == '__main__':
    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_DIR = os.path.join(args.log_dir, suffix)
    DESCS_DIR = os.path.join(LOG_DIR, 'temp_descs')
    if TEST_ON_W1BS:
        if not os.path.isdir(DESCS_DIR):
            os.makedirs(DESCS_DIR)
    logger, file_logger = None, None
    model = HardNet()
    if(args.enable_logging):
        from Loggers import Logger, FileLogger
        logger = Logger(LOG_DIR)
        #file_logger = FileLogger(./log/+suffix)
    #train_loader, test_loaders = create_loaders(load_random_triplets = triplet_flag)
    
    training_dataset = MegaDepthDataset(
    scene_list_path='/cluster/scratch/rthakur/dl/hardnet/code/train_scenes.txt',
    preprocessing='caffe',
    image_size=args.imageSize
)
    train_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        num_workers=4
    )
    validation_dataset = MegaDepthDataset(
        scene_list_path='/cluster/scratch/rthakur/dl/hardnet/code/valid_scenes.txt',
        train=False,
        preprocessing='caffe',
#        pairs_per_scene=25,
        image_size=args.imageSize
    )
    test_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=4
    )
    """
    print("before build")
    training_dataset.build_dataset()

    validation_dataset.build_dataset()
    print("after build")
    data=[]
    for i in range(16):
        print(i)
        data.append(validation_dataset.__getitem__(i))
    import pickle

    writefile = open("/cluster/scratch/tianhu/test/testgen_valid.pkl", "wb")
    pickle.dump(data, writefile)
    writefile.close()

    """
    """
    training_dataset.build_dataset()
    validation_dataset.build_dataset()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar:
        if batch_idx==1:
            break
    
        

        data_a, data_p = data
    pbar = tqdm(enumerate(test_loader))
    print("pbar",pbar)
    for batch_idx, data1 in pbar:
        if batch_idx==1:
            break
    import pickle 
    writefile = open("/cluster/scratch/tianhu/test/testgen_valid_1.pkl", "wb")
    pickle.dump([data,data1], writefile)
    writefile.close()
"""
    main(train_loader, test_loader,training_dataset, validation_dataset,model, logger, file_logger)
