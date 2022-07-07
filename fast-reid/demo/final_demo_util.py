import sys
sys.path.append('/home/ubuntu/Nilay/PersonReIDModels/LUPerson-extended/fast-reid')
from demo_visualizer import Visualizer
from predictor import FeatureExtractionDemo
from fastreid.config import get_cfg
from datasets import load_accumulated_info_of_dataset
import os
import numpy as np
import pandas as pd
from fastreid.utils.file_io import PathManager
from PIL import Image
import torchvision.transforms as T
from fastreid.data.transforms import *
import requests # to get image from the web

def read_image_url(image, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.
    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"
    Returns:
        image (np.ndarray): an HWC image
    """
    # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format == "BGR":
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    if format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)
    image = Image.fromarray(image)
    return image

def build_transforms(cfg):
    res = []
    size_test = cfg.INPUT.SIZE_TEST
    res.append(T.Resize(size_test, interpolation=3))
    res.append(ToTensor())
    return T.Compose(res)


def load_saved_feat(dir=None):
    assert os.path.isdir(dir), f"{dir} doesn't exist"
    feat_file = os.path.join(dir, "feat.npy")
    pid_file = os.path.join(dir, "pid.npy")
    camid_file = os.path.join(dir, "camid.npy")
    imgpath_file = os.path.join(dir, "imgpath.npy")

    assert os.path.isfile(feat_file), "Features not found"
    assert os.path.isfile(pid_file), "Pids not found"
    assert os.path.isfile(camid_file), "Camids not found"

    if os.path.isfile(imgpath_file):
        print("File containing saved image paths found")
        return (np.load(feat_file, allow_pickle=True).astype(np.float32),\
             np.load(pid_file, allow_pickle=True).astype(np.float32), np.load(camid_file, allow_pickle=True), np.load(imgpath_file, allow_pickle=True).tolist())    

    return (np.load(feat_file, allow_pickle=True).astype(np.float32),\
         np.load(pid_file, allow_pickle=True).astype(np.float32), np.load(camid_file, allow_pickle=True), None)


def get_params(dataset_name='LaST', trained_on_LaST=True):
    root_folder = "/home/ubuntu/Nilay/PersonReIDModels/Datasets"
    root_feat_dir = '/home/ubuntu/Nilay/PersonReIDModels/qualitative_evaluation/cache/LUPerson-extended'
    if trained_on_LaST:
        print("Loading model trained on LaST ...")
        config_file = "/home/ubuntu/Nilay/PersonReIDModels/LUPerson-extended/fast-reid/configs/LaST/mgn_R50_moco_cache_test.yml"
        feat_input_dir = os.path.join(root_feat_dir, "trained")
    else:
        print("Loading model trained on Market ...")
        config_file = "/home/ubuntu/Nilay/PersonReIDModels/LUPerson-extended/fast-reid/configs/CMDM/mgn_R50_moco.yml"
        feat_input_dir = root_feat_dir
    
    return config_file, root_folder, dataset_name, feat_input_dir


def set_up_demo(dataset_name='LaST', trained_on_LaST=True):
    config_file, root_folder, dataset_name, feat_input_dir = get_params(dataset_name, trained_on_LaST)
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.freeze()

    demo = FeatureExtractionDemo(cfg)

    _, query, gallery = load_accumulated_info_of_dataset(root_folder, dataset_name, use_eval_set=False, verbose=True)
    num_query = pd.DataFrame(query)["image_file_path"].nunique()
    test_dataset =  query + gallery

    feats, pids, camids, imgpaths = load_saved_feat(os.path.join(feat_input_dir, dataset_name))
    g_feats = feats[num_query:]
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])

    visualizer = Visualizer(test_dataset, None, None, num_query=num_query)
    visualizer.index_gallery(g_feats=g_feats, g_pids=g_pids, g_camids=g_camids)

    transform = build_transforms(cfg)

    return transform, demo, visualizer
