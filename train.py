import torch
import torch.nn as nn
from backbone.model_resnet import ResNet_50

# from tensorboardX import SummaryWriter
# import wandb
# from tqdm import tqdm
import os
from glob import glob

if __name__ == '__main__':
    # os.makedirs(os.path.join(MODEL_ROOT, EXP_NAME), exist_ok=True)
    # os.makedirs(LOG_ROOT, exist_ok=True)

    MODEL_ROOT = '.'
    EXP_NAME = '123'
    DEVICE = 'cuda:0'
    GPU_ID = [0]

    os.makedirs(os.path.join(MODEL_ROOT, EXP_NAME), exist_ok=True)

    # wandb.init(project=PROJECT_NAME, config=cfg)

    #======= model & loss & optimizer =======#
    BACKBONE = ResNet_50([112, 112])

    BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
    BACKBONE = BACKBONE.to(DEVICE)
        
    torch.save(BACKBONE.module.state_dict(), os.path.join(MODEL_ROOT, "Backbone_checkpoint.pth"))
