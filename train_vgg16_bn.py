import os
import sys
import argparse
import logging
import time
import datetime
import traceback

import torch
import torch.nn.functional as F 
import numpy as np 
from models.vgg import Vgg16_BN
from utils.network.Net import save_ckpt, load_weights
from models.Loss import CE_loss
from sklearn.metrics import log_loss
from utils.network.Init import init_random_seed
from datasets.DataFactory import get_data_loader 
from torch.nn.parallel._functions import Broadcast, ReduceAddCoalesced

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 手动设置gpu list
gpu_list = [0, 1, 2]

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a classifier network')

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='vgg16_bn', choices=['vgg16, vgg16_bn'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'val'])
    parser.add_argument('--category', type=str, default="heads_13_RetinaFace", choices=["heads_13_RetinaFace"])
    parser.add_argument('--landmarks', type=bool, default=False, choices=[True, False])

    # Path
    parser.add_argument('--save_dir', dest='save_dir', help='directory to save models', default="./output", type=str)
    parser.add_argument('--log_dir', dest='log_dir', help="directory to log", default='log', type=str)

    # Training settings
    parser.add_argument('--num_workers', dest='num_workers', help='number of worker to load data', default=8, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size', default=64, type=int)
    parser.add_argument('--max_iter', dest='max_iter', help='max_iter', default=30000, type=int)
    parser.add_argument('--lr', dest='lr', help='starting learning rate', type=float, default=0.0005)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', help='learning rate decay ratio', type=float, default=0.9)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step', help='learning rate decay step', type=int, default=1000)
    parser.add_argument('--o', dest='optimizer', help='Training optimizer.', default='SGD')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight decay for SGD')

    # Step size
    parser.add_argument('--disp_interval', dest='disp_interval', help='number of iterations to display', default=100, type=int)
    parser.add_argument('--save_interval', dest='save_interval', help='number of iterations to save', default=2000, type=int)
    parser.add_argument('--test_interval', dest='test_interval', help='number of iterations to test', default=2000, type=int)
    parser.add_argument('--checkpoint_start', dest='checkpoint_start', help='checkpoint to start load model', default=2000, type=int)
    parser.add_argument('--checkpoint_end', dest='checkpoint_end', help='checkpoint to start load model', default=30001, type=int)

    # Misc
    parser.add_argument('--seed', type=int, default=2020, help='random seed (default: 1)')
    parser.add_argument('--use_tensorboard', dest='use_tensorboard',
                        help='whether use tensorflow tensorboard',
                        default=True, type=bool)
    parser.add_argument('--mutil_gpu', type=bool, default=False, help='mutil GPU')
    parser.add_argument('--gpu_list', type=list, default=gpu_list, help='GPU list')
    return parser.parse_args()


class Solver(object):
    def __init__(self, config, dataloader):
        super(Solver, self).__init__()
        self.config = config 
        self.dataloader = dataloader 
        self.start_step = 0
        self.build_model()

        # Build tensorboard if use
        if self.config.use_tensorboard and self.config.mode == 'train':
            self.build_tensorboard()
    
    def build_tensorboard(self):
        from tensorboardX import SummaryWriter
        # Set the tensorboard logger
        self.tb_logger = SummaryWriter(self.config.log_dir)

    def build_model(self):
        if self.config.model == "vgg16_bn":
            self.model = Vgg16_BN()
        else:
            print("build model error: there is no %s" % (config.model))
            return None 
        assert self.model is not None 
        if self.config.mode == 'train':
            if self.config.optimizer == 'Adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))
            elif self.config.optimizer == "SGD":
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                lr=self.config.lr, momentum=self.config.momentum,
                                                weight_decay=self.config.weight_decay)
            # anneal
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                step_size=self.config.lr_decay_step,
                                                                gamma=self.config.lr_decay_gamma)
            
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            if self.config.mutil_gpu:
                '''
                torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
                module: 需要多GPU训练的网络模型
                device_ids: GPU的编号，（默认全部GPU， 或[0,1],[0,1,2]）
                output_device: (默认是device_ids[0])
                dim: tensor被分散的维度，默认是0
                '''
                self.model = torch.nn.DataParallel(self.model, self.config.gpu_list)
                

    def train(self):
        # setting to train mode
        self.model.train()
        start_time = time.time()
        data_iter = iter(self.dataloader)
        try:
            for step in range(self.start_step, self.config.max_iter):
                self.lr_scheduler.step()
                try:
                    img_batch, label_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    img_batch, label_batch = next(data_iter)
                if torch.cuda.is_available():
                        img_batch = img_batch.cuda()
                        label_batch = label_batch.cuda()
                logit = self.model(img_batch)
                total_loss = CE_loss(logit, label_batch)

                # backward
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # display results
                if (step + 1) % self.config.disp_interval == 0:
                    loss = {}
                    loss['S/total_loss'] = total_loss.item()
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    log = 'time cost: {} iter: {} / {}' \
                        .format(elapsed, step + 1, self.config.max_iter)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                
                    # Tensorboard logger
                    if self.config.use_tensorboard:
                        for tag, value in loss.items():
                            self.tb_logger.add_scalar(tag, value, step + 1)
                
                if (step + 1) % self.config.save_interval == 0:
                    state = {
                        'step':step + 1,
                        'optimizer': self.optimizer.state_dict(),
                        'model': self.model.module.state_dict()
                    }
                    save_ckpt(self.config.save_dir, state, step + 1)


        except(RuntimeError, KeyboardInterrupt):
            del data_iter
            stack_trace = traceback.format_exc()
            print(stack_trace)
        finally:
            if self.config.use_tensorboard:
                self.tb_logger.close()
    
    def test(self):
        self.model.eval()
        results = []
        for i in range(self.config.checkpoint_start, self.config.checkpoint_end, self.config.test_interval):
            checkpoint_path = os.path.join(self.config.save_dir, 'ckpt', 'model_' + str(i) + '.pth')
            self.model = load_weights(self.model, checkpoint_path)
            torch.cuda.empty_cache()
            
            all_scores = []
            all_labels = []
            torch_log_loss = 0
            for x_batch, label_batch in self.dataloader:
                if torch.cuda.is_available():
                    x_batch = x_batch.cuda()
                    label_batch = label_batch.cuda()

                    logits = self.model(x_batch)
                    logits = F.softmax(logits, dim=1)
                    torch_log_loss += CE_loss(logits, label_batch).detach().cpu() * label_batch.shape[0]
                    all_scores.extend(logits.detach().cpu().numpy().tolist())
                    all_labels.extend(label_batch.cpu().numpy().tolist())
            torch_log_loss /= len(all_labels)
            all_scores = np.array(all_scores)
            all_labels = np.array(all_labels)
            sk_log_loss = log_loss(all_labels, all_scores)
            result = 'Model:{},Total images: {}, sk_log_loss:{}, torch_log_loss:{}'.format(str(i), len(all_labels), sk_log_loss, torch_log_loss.item())
            print(result)
            results.append(result)

        with open(os.path.join(self.config.save_dir, 'results.txt'), 'w') as f:
            for item in results:
                f.write(item + ' \n')
        f.close()

        

def main():
    args = parse_args()
    logger.info('\t Called with args:')
    logger.info(args)

    if not torch.cuda.is_available():
        sys.exit('Need a CUDA device to run the code.')
    else:
        args.cuda = True
        if len(args.gpu_list) == 1:
            torch.cuda.set_device(args.gpu_list[0])
        else:
            args.mutil_gpu = True 

    init_random_seed(args.seed, args.cuda)

    save_dir = os.path.join(args.save_dir, args.model, args.category)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger.info('\t output will be saved to {}'.format(save_dir))

    log_dir = os.path.join(save_dir, args.log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger.info('\t logs will be saved to {}'.format(log_dir))

    args.save_dir = save_dir
    args.log_dir = log_dir

    data_root = '/root/data/DFDC/fb_dfd_release_0.1_final'

    train_dataloader = get_data_loader(data_root=data_root, 
                                        category=args.category, 
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers, 
                                        input_size=224, 
                                        interval=5,
                                        landmarks=args.landmarks, 
                                        mode='train')
    trainer = Solver(args, train_dataloader)
    trainer.train()

    test_dataloader = get_data_loader(data_root=data_root, 
                                        category=args.category, 
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers, 
                                        input_size=224, 
                                        interval=10,
                                        landmarks=args.landmarks, 
                                        mode='test')
    trainer = Solver(args, test_dataloader)
    trainer.test()


if __name__ == '__main__':
    main()
