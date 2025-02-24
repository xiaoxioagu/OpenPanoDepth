import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np

from utils.common import config_loader
from dataload import load_dataset, load_test_dataset
from base_trainer import Trainer
from base_tester import Tester

def arg_parse():
	parser = argparse.ArgumentParser(description='Main program for OpenPano.')
	parser.add_argument('--config', type=str, default='E:\python_project\OpenPanoDepth\configs\Bifusev2\Bifusev2_super_test.yaml',
						help="path of config file")
	# parser.add_argument('--config', type=str, default='E:\python_project\OpenPanoDepth\configs\Depth_Anything\Depth_Anything_test.yaml',
	# 						help="path of config file")
	parser.add_argument('--scope', default='train', choices=['train', 'val', 'test'],
						help="choose train or test scope")
	parser.add_argument('--has_pth', default='True',help="have model")
	parser.add_argument('--device', type=str, default='cuda', help="device to use for non-distributed mode.")
	parser.add_argument("--gpu_devices", type=int, nargs="+", default=[0], help="available gpus")
	parser.add_argument('--seed', type=int, default=42, metavar='S',
						help='random seed (default: 1)')
	opt = parser.parse_args()
	return opt

def worker(opt,cfgs):
	random.seed(opt.seed)
	np.random.seed(opt.seed)
	torch.manual_seed(opt.seed)
	torch.cuda.manual_seed_all(opt.seed)

	model_cfg = cfgs['model_cfg']
	data_cfg = cfgs['data_cfg']
	worker_cfg = cfgs['worker_cfg']
	scope = opt.scope


	if scope == 'train':
		# 加载数据集
		train_dataloader, val_dataloader = load_dataset(data_cfg)
		# 加载模型
		trainer = Trainer(model_cfg, worker_cfg, opt)
		trainer.train(train_dataloader, val_dataloader)

	elif scope == 'test':
		test_dataloader = load_test_dataset(data_cfg)
		tester = Tester(model_cfg, worker_cfg, opt)
		tester.test(test_dataloader)

if __name__ == '__main__':
	opt = arg_parse()
	cfgs = config_loader(opt.config)
	worker(opt,cfgs)