import torch
import os
import tqdm

from modeling import Omnifusion_model, Unifuse_model, Bifuse_model, HoHonet_model,\
	                 OmniDepth_RectNet_model,OmniDepth_UResNet_model,PanoFormer_model,\
                     SliceNet_model,Bifusev2_self_model,Bifusev2_super_model,\
                     Svsyn_model,Joint_360Depth_model,GLPanoDepth_model,ACDNet_model,\
                     HRDFuse_model,EGformer_model,\
	                 Bifusev3_model,Bifusev4_model,Bifusev3_equi_model,\
                     IGEV_Bifuse_model,Bifusev3_tp_model,IGEV_Bifuse_cp_model,ACDNet_Bifuse_model,Depth_Anything_model

from evaluation.metrics import compute_depth_metrics, Evaluator
from utils.saver import Saver

class Tester:
	def __init__(self, model_cfg,test_cfg, opt):
		self.model_cfg = model_cfg
		self.test_cfg = test_cfg
		self.opt = opt

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.load_weights_folder = os.path.expanduser(self.model_cfg["load_checkpoint"])
		model_path = os.path.join(self.load_weights_folder)
		model_dict = torch.load(model_path, map_location=self.device)

		Net_dict = {"Omnifusion": Omnifusion_model,
                    "Unifuse": Unifuse_model,
                    "Bifuse": Bifuse_model,
                    "HoHonet": HoHonet_model,
                    "RectNet": OmniDepth_RectNet_model,
                    "UResNet": OmniDepth_UResNet_model,
                    "PanoFormer": PanoFormer_model,
                    "SliceNet": SliceNet_model,
                    "Bifusev2_self": Bifusev2_self_model,
                    "Bifusev2_super": Bifusev2_super_model,
                    "Svsyn": Svsyn_model,
                    "Joint_360Depth": Joint_360Depth_model,
                    "GLPanoDepth": GLPanoDepth_model,
                    "ACDNet": ACDNet_model,
                    "HRDFuse": HRDFuse_model,
                    "EGformer": EGformer_model,
					"Bifusev3": Bifusev3_model,
					"Bifusev4": Bifusev4_model,
					"Bifusev3_equi": Bifusev3_equi_model,
					"IGEV_Bifuse": IGEV_Bifuse_model,
					"Bifusev3_tp": Bifusev3_tp_model,
					'IGEV_Bifuse_cp': IGEV_Bifuse_cp_model,
					'ACDNet_Bifuse': ACDNet_Bifuse_model,
					'Depth_Anything': Depth_Anything_model}
		Net = Net_dict[model_cfg["model"]]
		self.model = Net(model_cfg["base_config"])

		self.model.to(self.device)
		self.model_state_dict = self.model.state_dict()
		self.model.load_state_dict({k: v for k, v in model_dict.items() if k in self.model_state_dict})

		self.evaluator = Evaluator()
		self.saver = Saver(os.path.join(self.test_cfg["save_path"], "test"))


	def test(self,test_dataloader):
		self.test_dataloader = test_dataloader
		self.model.eval()
		pbar = tqdm.tqdm(self.test_dataloader)
		for batch_idx, (rgb, depth, mask) in enumerate(pbar):
			rgb, depth, mask = rgb.cuda(), depth.cuda(), mask.cuda()

			with torch.no_grad():
				equi_outputs_list = self.model(rgb)
				equi_outputs = equi_outputs_list
				if self.model_cfg["model"] == "IGEV_Bifuse" or self.model_cfg["model"] == "IGEV_Bifuse_cp":
					equi_outputs = equi_outputs[2]
				error = torch.abs(depth - equi_outputs) * mask
				error[error < 0.1] = 0

				self.evaluator.compute_eval_metrics(equi_outputs, depth, mask)
				if batch_idx % 1 == 0:
					self.saver.save_samples(rgb, depth, equi_outputs, mask)
		self.evaluator.print(self.test_cfg["save_path"])

