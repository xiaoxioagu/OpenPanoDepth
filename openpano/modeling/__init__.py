from .Omnifusion.omnifusion.spherical_model_iterative import spherical_fusion as Omnifusion_model
from .Unifuse.unifuse.unifuse import UniFuse as Unifuse_model
from .Bifuse.bifuse.FCRN import MyModel as Bifuse_model
from .HoHonet.hohonet.hohonet import HoHoNet as HoHonet_model
from .OmniDepth.omnidepth.network import RectNet as OmniDepth_RectNet_model
from .OmniDepth.omnidepth.network import UResNet as OmniDepth_UResNet_model
from .PanoFormer.panoformer.model import Panoformer as PanoFormer_model
from .SliceNet.slicenet.slice_model import SliceNet as SliceNet_model
from .Bifusev2.bifusev2.BiFuse import SupervisedCombinedModel as Bifusev2_super_model
from .Bifusev2.bifusev2.BiFuse import SelfSupervisedCombinedModel as Bifusev2_self_model
from .Svsyn.svsyn.resnet360 import ResNet360 as Svsyn_model
from .Joint_360Depth.joint_360depth.models import DPTDepthModel as Joint_360Depth_model
from .GLPanoDepth.glpanodepth.TwoBranch import TwoBranch as GLPanoDepth_model
from .ACDNet.acdnet.acdnet.acdnet import ACDNet as ACDNet_model
from .HRDFuse.hrdfuse.HRDFuse import hrdfuse as HRDFuse_model
from .EGformer.egformer.egformer import EGTransformer as EGformer_model

from .Bifusev3.bifusev3.BiFuse import SupervisedCombinedModel as Bifusev3_model
from .Bifusev4.bifusev4.BiFuse import SupervisedCombinedModel as Bifusev4_model
from .Bifusev5.bifusev5.BiFuse import SupervisedCombinedModel as Bifusev5_model
from .Bifusev3.bifusev3.BiFuse import SupervisedCombinedModel_equi as Bifusev3_equi_model
from .Bifusev3.bifusev3.BiFuse import SupervisedCombinedModel_tp as Bifusev3_tp_model
from .IGEV_Bifuse.IGEV_bifuse.igev_pano import IGEVPano as IGEV_Bifuse_model
from .ACDNet_Bifuse.ACDNet_bifuse.acd_bifuse import ACDNet_Bifuse as ACDNet_Bifuse_model
from .MK_Bifuse.mk_bifuse.igev_pano import IGEVPano as MK_Bifuse_model
from .IGEV_Bifuse_cp.IGEV_bifuse.igev_pano import IGEVPano as IGEV_Bifuse_cp_model
from .Depth_Anything.depth_anything.dpt import DPT_DINOv2 as Depth_Anything_model

