# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_da_config(cfg):
    """
    Add config for DA training.
    """
    cfg.CUDNN_BENCHMARK = True
    cfg.MODEL.DIS_TYPE = "vgg4"
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.STEPS = (50000,)
    cfg.SOLVER.MAX_ITER = 70000
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.DATASETS.TRAIN = ("my_cityscapes_train",)
    cfg.DATASETS.TRAIN_TARGETS = ("my_foggy_cityscapes_train",)
    cfg.DATASETS.TEST = ("my_foggy_cityscapes_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.INPUT.MIN_SIZE_TEST = 600
    cfg.INPUT.MIN_SIZE_TRAIN = (600,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1200
    cfg.INPUT.MAX_SIZE_TEST = 1200
    cfg.OUTPUT_DIR = "./runs/train-da-faster1"
    
def add_so_config(cfg):
    """
    Add config for SO training.
    """
    cfg.CUDNN_BENCHMARK = True
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.STEPS = (50000,)
    cfg.SOLVER.MAX_ITER = 70000
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.DATASETS.TRAIN = ("my_cityscapes_train",)
    cfg.DATASETS.TEST = ("my_foggy_cityscapes_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.INPUT.MIN_SIZE_TEST = 600
    cfg.INPUT.MIN_SIZE_TRAIN = (600,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1200
    cfg.INPUT.MAX_SIZE_TEST = 1200
    cfg.OUTPUT_DIR = "./runs/train-so-faster1"