import glob
import os
import numpy as np

import pandas as pd


def _get_pid2label(dir_path):
    img_paths = glob.glob(os.path.join(dir_path, '*/*.jpg'))
    pid_container = set()
    for img_path in img_paths:
        pid = int(os.path.basename(img_path).split('_')[0])
        pid_container.add(pid)
    pid_container = np.sort(list(pid_container))
    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    return pid2label


def _process_dir(dir_path, pid2label=None, relabel=False, recam=0):
    if 'query' in dir_path:
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
    else:
        img_paths = glob.glob(os.path.join(dir_path, '*/*.jpg'))
    img_paths = sorted(img_paths)
    accumulated_info_list = []
    for ii, img_path in enumerate(img_paths):
        pid = int(os.path.basename(img_path).split('_')[0])
        camid = int(recam + ii)
        if relabel and pid2label is not None:
            pid = pid2label[pid]
        accumulated_info = {
            "image_file_path": img_path,
            "identity_ID": pid,
            "camera_ID": camid
        }
        accumulated_info_list.append(accumulated_info)
    return accumulated_info_list


def _load_accumulated_info(root_folder_path,
                           dataset_folder_name="last",
                           image_folder_name="test", 
                           subfolder_name = "gallery", recam=0):
    """LaST.

    Reference:
        LaST: Large-Scale Spatio-Temporal Person Re-identification

    URL: `<https://github.com/shuxjweb/last#last-large-scale-spatio-temporal-person-re-identification>`_

    Dataset statistics:
        LaST is a large-scale dataset with more than 228k pedestrian images. 
        It is used to study the scenario that pedestrians have a large activity scope and time span. 
        Although collected from movies, we have selected suitable frames and labeled them as carefully as possible. 
        Besides the identity label, we also labeled the clothes of pedestrians in the training set.
        
        Train: 5000 identities and 71,248 images.
        Val: 56 identities and 21,379 images.
        Test: 5806 identities and 135,529 images.
        --------------------------------------
        subset         | # ids     | # images
        --------------------------------------
        train          |  5000     |    71248
        query          |    56     |      100
        gallery        |    56     |    21279
        query_test     |  5805     |    10176
        gallery_test   |  5806     |   125353
    """
    dataset_folder_path = os.path.join(root_folder_path, dataset_folder_name)

    if subfolder_name is not None:
        image_folder_path = os.path.join(dataset_folder_path, image_folder_name, subfolder_name)
    else:
        image_folder_path = os.path.join(dataset_folder_path, image_folder_name)

    if image_folder_name == "train":
        pid2label = _get_pid2label(image_folder_path)
        accumulated_info_list = _process_dir(image_folder_path, pid2label=pid2label, relabel=True)
    else:
        accumulated_info_list = _process_dir(image_folder_path, relabel=False, recam=recam)

    return accumulated_info_list


def _get_imagedata_info(datalist: list):
    data = pd.DataFrame(datalist)    
    num_pids = data["identity_ID"].nunique()
    num_cams = data["camera_ID"].nunique()
    num_imgs = data["image_file_path"].nunique()
    return num_pids, num_imgs, num_cams

  
def _print_dataset_statistics_movie(train, query, gallery, use_eval_set):
    num_train_pids, num_train_imgs, _ = _get_imagedata_info(train)
    num_query_pids, num_query_imgs, _ = _get_imagedata_info(query)
    num_gallery_pids, num_gallery_imgs, _ = _get_imagedata_info(gallery)

    if use_eval_set:
        test_or_eval = "eval"
    else:
        test_or_eval = "test"

    print("Dataset statistics:")
    print("  --------------------------------------")
    print("  subset         | # ids     | # images")
    print("  --------------------------------------")
    print("  train          | {:5d}     | {:8d}".format(num_train_pids, num_train_imgs))
    print("  query   ({})       | {:5d}     | {:8d}".format(test_or_eval, num_query_pids, num_query_imgs))
    print("  gallery ({})      | {:5d}     | {:8d}".format(test_or_eval, num_gallery_pids, num_gallery_imgs))

def load_LaST(root_folder_path, use_eval_set=False, verbose=True, **kwargs):
    test_folder = "test"
    if use_eval_set:
        test_folder = "val"
    train_and_valid_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path,
        image_folder_name="train", subfolder_name=None)
    test_query_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path, image_folder_name=test_folder, subfolder_name="query")
    test_gallery_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path,
        image_folder_name=test_folder, subfolder_name="gallery", recam=len(test_query_accumulated_info_dataframe))

    if verbose:
        _print_dataset_statistics_movie(train_and_valid_accumulated_info_dataframe,\
             test_query_accumulated_info_dataframe, test_gallery_accumulated_info_dataframe, use_eval_set)

    return train_and_valid_accumulated_info_dataframe, test_query_accumulated_info_dataframe, test_gallery_accumulated_info_dataframe
