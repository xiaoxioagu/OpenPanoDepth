from .dataset_loader_matterport import Dataset as matterport
from .dataset_loader_stanford import Dataset as stanford
from .dataset_loader_360d import Dataset as threed60

import torch

def load_dataset(data_cfg):
	datasets_dict = {"3d60": threed60,
					 "stanford2d3d": stanford,
					 "matterport3d": matterport}
	train_dataset = datasets_dict[data_cfg["name"]]
	val_dataset = datasets_dict[data_cfg["name"]]

	train_dataloader = torch.utils.data.DataLoader(
		dataset=train_dataset(
			rotate=True,
			flip=True,
			root_path= data_cfg["input_dir"],
			path_to_img_list= data_cfg["train_file"],
		    type = data_cfg["type"]),
		batch_size= data_cfg["train_batch_size"],
		shuffle= data_cfg["shuffle"],
		num_workers= data_cfg["num_workers"],
		drop_last=True)

	val_dataloader = torch.utils.data.DataLoader(
		dataset=val_dataset(
			root_path= data_cfg["input_dir"],
			path_to_img_list= data_cfg["test_file"],
		    type = data_cfg["type"]),
		batch_size= data_cfg["test_batch_size"],
		shuffle=False,
		num_workers= data_cfg["num_workers"],
		drop_last=True)

	return train_dataloader, val_dataloader

def load_test_dataset(data_cfg):
	datasets_dict = {"3d60": threed60,
					 "stanford2d3d": stanford,
					 "matterport3d": matterport}
	test_dataset = datasets_dict[data_cfg["name"]]
	test_dataloader = torch.utils.data.DataLoader(
		dataset=test_dataset(
			root_path=data_cfg["input_dir"],
			path_to_img_list=data_cfg["test_file"],
		    type = data_cfg["type"]),
		batch_size=data_cfg["test_batch_size"],
		shuffle=False,
		num_workers=data_cfg["num_workers"],
		drop_last=True)
	return test_dataloader





