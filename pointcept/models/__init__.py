from .builder import build_model
from .default import DefaultSegmentor, DefaultClassifier
from .modules import PointModule, PointModel

# Backbones
from .sparse_unet import *
from .point_transformer import *
from .point_transformer_v2 import *
from .point_transformer_v3 import *
from .stratified_transformer import *
from .spvcnn import *
from .octformer import *
from .oacnns import *

# from .swin3d import *

# Semantic Segmentation
from .context_aware_classifier import *

# Instance Segmentation
from .point_group import *

# Pretraining
from .masked_scene_contrast import *
from .point_prompt_training import *
from .sonata import *

# interactive segmentation, 导入, 确保 registrition 成功
from .default_iseg import DefaultInteractiveSegmentor, DefaultISegEncoder
from .mask3d import *
from .interobj import *
from .agile3d import *
