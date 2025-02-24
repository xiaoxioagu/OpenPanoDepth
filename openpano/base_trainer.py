import torch
import os
import torch.optim as optim
import time
import tqdm

from evaluation.metrics import compute_depth_metrics, Evaluator
from utils.saver import Saver
from utils.loss import compute_berhu_loss,compute_igev_loss

from modeling import Omnifusion_model, Unifuse_model,Bifuse_model,HoHonet_model,\
                     OmniDepth_RectNet_model,OmniDepth_UResNet_model,PanoFormer_model,\
                     SliceNet_model,Bifusev2_self_model,Bifusev2_super_model,\
                     Svsyn_model,Joint_360Depth_model,GLPanoDepth_model,ACDNet_model,\
                     HRDFuse_model,EGformer_model,\
                     Bifusev3_model,Bifusev4_model,Bifusev5_model,Bifusev3_equi_model,\
                     IGEV_Bifuse_model,Bifusev3_tp_model,ACDNet_Bifuse_model,MK_Bifuse_model,IGEV_Bifuse_cp_model


class Trainer:
    def __init__(self, model_cfg,train_cfg, opt):
        self.opt = opt
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        self.best_train_loss = float("inf")

        self.device = torch.device("cuda" if opt.device == "cuda" else "cpu")
        self.gpu_devices = ','.join([str(id) for id in opt.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices

        # network
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
                    "Bifusev5": Bifusev5_model,
                    "Bifusev3_equi": Bifusev3_equi_model,
                    "Bifusev3_tp": Bifusev3_tp_model,
                    "IGEV_Bifuse": IGEV_Bifuse_model,
                    "ACDNet_Bifuse": ACDNet_Bifuse_model,
                    "MK_Bifuse": MK_Bifuse_model,
                    "IGEV_Bifuse_cp": IGEV_Bifuse_cp_model}
        Net = Net_dict[model_cfg["model"]]

        self.model = Net(model_cfg["base_config"])

        self.model.to(self.device)
        self.parameters_to_train = list(self.model.parameters())

        self.optimizer = optim.AdamW(self.parameters_to_train, lr= self.train_cfg["optimizer_cfg"]["lr"])

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=1e-6,
                                                                   last_epoch=-1)

        if self.model_cfg["load_checkpoint"] != "None":
            self.load_model()

        print("Training model named:\n ", self.model_cfg["model"])
        print("Models and tensorboard events files are saved to:\n", self.train_cfg["save_path"])
        print("Training is using:\n ", self.device)

        self.evaluator = Evaluator()
        self.saver = Saver(os.path.join(self.train_cfg["save_path"], "val"))

        if not os.path.isdir(os.path.join(self.train_cfg["save_checkpoint"])):
            os.makedirs(os.path.join(self.train_cfg["save_checkpoint"]))

    def load_model(self):
        self.model_cfg["load_checkpoint"] = os.path.expanduser(self.model_cfg["load_checkpoint"])
        # assert os.path.isdir(self.model_cfg["load_checkpoint"]), \
        #     "Cannot find folder {}".format(self.model_cfg["load_checkpoint"])
        print("loading model from folder {}".format(self.model_cfg["load_checkpoint"]))

        path = self.model_cfg["load_checkpoint"]
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def train(self, train_loader, val_loader):
        self.epoch = 0
        self.step = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.start_full_time = time.time()
        for self.epoch in range(self.train_cfg["epochs"]):
            self.train_one_epoch()
            if self.epoch % 5 == 0:
                self.validate()
        print("Training Ended")
        print('full training time = %.2f HR' % ((time.time() - self.start_full_time) / 3600))

    def train_one_epoch(self):
        self.model.train()

        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(self.epoch))
        self.total_train_loss = 0
        self.total_depth_loss = 0

        for batch_idx, inputs in enumerate(pbar):

            outputs, losses = self.process_batch(inputs)
            loss = losses

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.total_train_loss += loss.item()
            self.total_depth_loss += losses.item()
            if batch_idx % self.train_cfg["visualize_interval"] == 0 and batch_idx > 0:
                train_loss = self.total_depth_loss / (batch_idx + 1)
                print('[Epoch %d--Iter %d]depth loss %.4f' %
                      (self.epoch, batch_idx, train_loss ))

        print('lr for epoch ', self.epoch, ' ', self.optimizer.param_groups[0]['lr'])
        torch.save(self.model.state_dict(),
                   self.train_cfg["save_checkpoint"] + '/checkpoint_latest.pth')

        if train_loss < self.best_train_loss:
             self.best_train_loss = train_loss
             torch.save(self.model.state_dict(),self.train_cfg["save_checkpoint"] + '/checkpoint_best.pth')

        self.scheduler.step()

    def process_batch(self, inputs):
        rgb = inputs[0]
        depth = inputs[1]
        mask = inputs[2]


        rgb = rgb.cuda()
        depth = depth.cuda()
        mask = mask.cuda()

        outputs = self.model(rgb)

        if self.model_cfg["model"] == "IGEV_Bifuse" or self.model_cfg["model"] == "MK_Bifuse" or self.model_cfg["model"] == "IGEV_Bifuse_cp":
            loss = compute_igev_loss(outputs, depth, mask)
        else:
            loss = compute_berhu_loss(outputs, depth, mask)

        return outputs, loss

    def validate(self):
        print('-------------Validate Epoch', self.epoch, '-----------')
        self.model.eval()
        pbar = tqdm.tqdm(self.val_loader)
        for batch_idx, (rgb, depth, mask) in enumerate(pbar):
            rgb, depth, mask = rgb.cuda(), depth.cuda(), mask.cuda()
            with torch.no_grad():
                equi_outputs = self.model(rgb)
                if self.model_cfg["model"] == "IGEV_Bifuse" or self.model_cfg["model"] == "MK_Bifuse":
                    equi_outputs = equi_outputs[-1]
                # error = torch.abs(depth - equi_outputs) * mask
                # error[error < 0.1] = 0

            if batch_idx % 20 == 0:
                self.saver.save_samples(rgb, depth, equi_outputs, mask)

            self.evaluator.compute_eval_metrics(equi_outputs, depth, mask)

        self.evaluator.print(self.train_cfg["save_path"])
        self.evaluator.reset_eval_metrics()


