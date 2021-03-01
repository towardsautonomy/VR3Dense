import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('src/utils')
sys.path.append('src/models')
sys.path.append('src/datasets')
sys.path.append('src/AB3DMOT')
sys.path.append('src/AB3DMOT/AB3DMOT_libs')
from .config import parse_args
from .trainer import Trainer
from models import VR3Dense
from models.loss_func import *
from AB3DMOT import AB3DMOT_libs