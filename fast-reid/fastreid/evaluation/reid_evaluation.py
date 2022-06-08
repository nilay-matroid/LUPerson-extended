# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
from collections import OrderedDict
from time import time
from sklearn import metrics

import numpy as np
import torch
import torch.nn.functional as F

from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank import evaluate_rank
from .rerank import re_ranking
from .roc import evaluate_roc
from fastreid.utils import comm
import os
import shutil
from npy_append_array import NpyAppendArray

logger = logging.getLogger(__name__)


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.use_cache = self.cfg.TEST.CACHE.ENABLED
        self.cache_dir = self.cfg.TEST.CACHE.CACHE_DIR
        self.parallelized = self.cfg.TEST.CACHE.PARALLEL.ENABLED
        self.num_workers = self.cfg.TEST.CACHE.PARALLEL.NUM_WORKERS

        if self.use_cache:
            assert self.cache_dir is not None
            if not os.path.isdir(self.cache_dir):
                os.mkdir(self.cache_dir)
            self.pid_file = os.path.join(self.cache_dir, "pid.npy")
            self.camid_file = os.path.join(self.cache_dir, "camid.npy")
            self.feat_file = os.path.join(self.cache_dir, "feat.npy")
            
        self.features = []
        self.pids = []
        self.camids = []

    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []

        if self.use_cache:
            # Delete the cache contents
            logger.info("Resetting ... Deleting any features which were cached")
            shutil.rmtree(self.cache_dir, ignore_errors=True)

            # Recreate the empty folder
            os.mkdir(self.cache_dir)


    def process(self, inputs, outputs):
        if self.use_cache:
            NpyAppendArray(self.pid_file).append(inputs["targets"].numpy())
            NpyAppendArray(self.camid_file).append(inputs["camids"].numpy())
            NpyAppendArray(self.feat_file).append(outputs.cpu().numpy())
        else:
            self.pids.extend(inputs["targets"])
            self.camids.extend(inputs["camids"])
            self.features.append(outputs.cpu())

    @staticmethod
    def cal_dist(metric: str, query_feat: torch.tensor, gallery_feat: torch.tensor):
        assert metric in ["cosine", "euclidean"], "must choose from [cosine, euclidean], but got {}".format(metric)
        if metric == "cosine":
            dist = 1 - torch.mm(query_feat, gallery_feat.t())
        else:
            m, n = query_feat.size(0), gallery_feat.size(0)
            xx = torch.pow(query_feat, 2).sum(1, keepdim=True).expand(m, n)
            yy = torch.pow(gallery_feat, 2).sum(1, keepdim=True).expand(n, m).t()
            dist = xx + yy
            dist.addmm_(query_feat, gallery_feat.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist.cpu().numpy()

    def check_cache(self):
        assert self.use_cache
        assert self.cache_dir is not None
        assert os.path.isdir(self.cache_dir)
        assert os.path.isfile(self.pid_file)
        assert os.path.isfile(self.camid_file)
        assert os.path.isfile(self.feat_file)

    def evaluate(self):
        if self.use_cache:
            # Not sure if multiprocessing can be done with current cache method
            assert comm.get_world_size() <= 1
            features = torch.tensor(np.load(self.feat_file, mmap_mode='r'))
            pids = torch.tensor(np.load(self.pid_file, mmap_mode='r'))
            camids = torch.tensor(np.load(self.camid_file, mmap_mode='r'))
            print("Feature size: ", features.shape)
            print("pids size: ", pids.shape)
            print("camids size: ", camids.shape)
        elif comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            pids = comm.gather(self.pids)
            pids = sum(pids, [])

            camids = comm.gather(self.camids)
            camids = sum(camids, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            features = self.features
            pids = self.pids
            camids = self.camids

        if not self.use_cache:
            features = torch.cat(features, dim=0)

        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])
        query_camids = np.asarray(camids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])
        gallery_camids = np.asarray(camids[self._num_query:])

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        if self.cfg.TEST.METRIC == "cosine":
            query_features = F.normalize(query_features, dim=1)
            gallery_features = F.normalize(gallery_features, dim=1)

        # TODO(nilay.pande): remove this permanently.
        # No Need of dist calculation which will only slow things
        # dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, gallery_features)
        dist = None

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            print("Calculating distance here instead now")
            start = time()
            dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, gallery_features)
            print("Time to calcuate initial distance mat: ", time() - start)
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA
            q_q_dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, query_features)
            g_g_dist = self.cal_dist(self.cfg.TEST.METRIC, gallery_features, gallery_features)
            re_dist = re_ranking(dist, q_q_dist, g_g_dist, k1, k2, lambda_value)

            print("Time for extraneous re-ranking computation: ", time() - start)
            query_features = query_features.numpy()
            gallery_features = gallery_features.numpy()
            cmc, all_AP, all_INP = evaluate_rank(re_dist, query_features, gallery_features,
                                                 query_pids, gallery_pids, query_camids,
                                                 gallery_camids, use_distmat=True, parallelize=self.parallelized, num_workers=self.num_workers)
            print("Time taken to evaluate rank with re-ranking computation: ", time() - start)
        else:
            query_features = query_features.numpy()
            gallery_features = gallery_features.numpy()
            start = time()
            cmc, all_AP, all_INP = evaluate_rank(dist, query_features, gallery_features,
                                                 query_pids, gallery_pids, query_camids, gallery_camids,
                                                 use_distmat=False, parallelize=self.parallelized, num_workers=self.num_workers)
            print("Time taken to evaluate rank without re-ranking computation: ", time() - start)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        if self.cfg.TEST.ROC_ENABLED:
            scores, labels = evaluate_roc(dist, query_features, gallery_features,
                                          query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)
