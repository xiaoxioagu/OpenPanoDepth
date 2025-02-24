import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
import cv2
import matplotlib.pyplot as plt

from modeling import Bifusev2_self_model

def only_picture(path,model,device,result_path):
	rgb = cv2.imread(path)
	base_name = os.path.basename(path)  # 获取原始文件名
	name, _ = os.path.splitext(base_name)  # 去掉扩展名
	output_path = f"{name}_depth.png"
	rgb = cv2.resize(rgb, (1024, 512), interpolation=cv2.INTER_AREA)
	rgb = rgb.astype(np.float32) / 255

	rgb = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0)
	rgb = rgb.to(device)
	model.eval()

	with torch.no_grad():
		depth = model(rgb)
	output_path = os.path.join(result_path, output_path)
	depth = depth.squeeze().cpu().numpy()
	cmap = plt.get_cmap("rainbow_r")
	depth = cmap(depth.astype(np.float32) / 10)
	depth = np.delete(depth, 3, 2)
	depth = (depth * 255).astype(np.uint8)
	cv2.imwrite(output_path, depth)

	return depth

def dataset(path,model,device,result_path):
	result = []
	for filename in os.listdir(path):
		if filename.endswith('.jpg') or filename.endswith('.png'):
			img_path = os.path.join(path, filename)
			depth = only_picture(img_path,model,device,result_path)
			result.append(depth)

	return result

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='demo for image depth estimation for OpenPano.')
	parser.add_argument('--model_path', type=str,
						default='E:\\python_project\\OpenPanoDepth\\result\\Bifusev2_self\\checkpoints\\checkpoint_best.pth',
						help="path of model file")
	parser.add_argument('--picture_path', type=str, default='E:\\python_project\\OpenPanoDepth\\picture\\rgb_coldlight.png',
						help="path of picture file")
	parser.add_argument('--result_path', type=str,
						default='E:\\python_project\\OpenPanoDepth\\picture',
						help="path of result")
	parser.add_argument('--num_picture', default='only', choices=['only', 'dataset'],
						help="choose only_picture or dataset")
	parser.add_argument('--color', default='True', help="the result is color or gray")
	parser.add_argument('--device', type=str, default='cuda', help="device to use for non-distributed mode.")
	parser.add_argument("--gpu_devices", type=int, nargs="+", default=[0], help="available gpus")
	parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
	opt = parser.parse_args()

	random.seed(opt.seed)
	np.random.seed(opt.seed)
	torch.manual_seed(opt.seed)
	torch.cuda.manual_seed_all(opt.seed)

	device = torch.device(opt.device)
	model_path = os.path.join(opt.model_path)
	model_dict = torch.load(model_path, map_location=device)

	num_picture = opt.num_picture
	color = opt.color
	picture_path = opt.picture_path
	result_path = opt.result_path

	bifusev2_model_cfg = {
		'save_path': 'E:\\python_project\\OpenPanoDepth\\result\\Bifusev2_super',
	    'dnet_args':{
			'layers': 34,
			'CE_equi_h': [8, 16, 32, 64, 128, 256, 512]
		},
		'pnet_args':{
			'layers': 18,
			'nb_tgts': 2
		}
	}

	model = Bifusev2_self_model(bifusev2_model_cfg).to(device)
	model_state_dict = model.state_dict()
	model.load_state_dict({k: v for k, v in model_dict.items() if k in model_dict})
	if num_picture == 'only':
		depth = only_picture(picture_path, model, device,result_path)
	else:
		depth = dataset(picture_path, model, device,result_path)


