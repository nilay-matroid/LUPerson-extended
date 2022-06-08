# encoding: utf-8
"""
@author:  Nilay Pande
@contact: nilay017@gmail.com
"""

import glob
import os.path as osp
import os
import numpy as np

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

__all__ = ['LaST', ]

@DATASET_REGISTRY.register()
class LaST(ImageDataset):
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
    dataset_dir = ''
    dataset_name = "last"

    def __init__(self, root='datasets', train_mode=True, verbose=False, **kwargs):
        """
        Train mode -> is_train = True
        Test_mode -> is_train = False
        """
        
        self.root = root
        self.is_train = train_mode
        self.dataset_dir = osp.join(self.root, self.dataset_name)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'val', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'val', 'gallery')
        self.query_test_dir = osp.join(self.dataset_dir, 'test', 'query')
        self.gallery_test_dir = osp.join(self.dataset_dir, 'test', 'gallery')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.query_test_dir,
            self.gallery_test_dir
        ]

        self.check_before_run(required_files)

        self.pid2label = self.get_pid2label(self.train_dir)
        train = self.process_dir(self.train_dir, pid2label=self.pid2label, relabel=True)

        if self.is_train:
            query = self.process_dir(self.query_dir, relabel=False)
            gallery = self.process_dir(self.gallery_dir, relabel=False, recam=len(query))
        else:
            query = self.process_dir(self.query_test_dir, relabel=False)
            gallery = self.process_dir(self.gallery_test_dir, relabel=False, recam=len(query))

        if verbose:
            print("=> LaST loaded")
            print("=> Train mode: ", self.is_train)
            self.print_dataset_statistics_movie(train, query, gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(gallery)

        super(LaST, self).__init__(train, query, gallery, **kwargs)


    def get_pid2label(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))            # [103367,]

        pid_container = set()
        for img_path in img_paths:
            pid = int(os.path.basename(img_path).split('_')[0])
            pid_container.add(pid)
        pid_container = np.sort(list(pid_container))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label


    def process_dir(self, dir_path, pid2label=None, relabel=False, recam=0):
        if 'query' in dir_path:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        else:
            img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))

        img_paths = sorted(img_paths)
        dataset = []
        for ii, img_path in enumerate(img_paths):
            pid = int(os.path.basename(img_path).split('_')[0])
            camid = int(recam + ii)
            if relabel and pid2label is not None:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def get_videodata_info(self, data, return_tracklet_info=False):
        pids, cams, tracklet_info = [], [], []
        for img_paths, pid, camid in data:
            pids += [pid]
            cams += [camid]
            tracklet_info += [len(img_paths)]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_tracklets = len(data)
        if return_tracklet_info:
            return num_pids, num_tracklets, num_cams, tracklet_info
        return num_pids, num_tracklets, num_cams

    def print_dataset_statistics_movie(self, train, query, gallery):
        num_train_pids, num_train_imgs, _ = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, _ = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, _ = self.get_imagedata_info(gallery)

        if self.is_train:
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
