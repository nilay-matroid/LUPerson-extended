# credits: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/metrics/rank.py

from time import time
import warnings
from collections import defaultdict

import faiss
import numpy as np
from tqdm import tqdm
import os
import pickle as pkl
import shutil

try:
    from .rank_cylib.rank_cy import evaluate_cy

    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython rank evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )


def eval_cuhk03(distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank, use_distmat):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed num_repeats times.
    """
    num_repeats = 10

    num_q, num_g = distmat.shape
    dim = q_feats.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(g_feats)
    if use_distmat:
        indices = np.argsort(distmat, axis=1)
    else:
        _, indices = index.search(q_feats, k=num_g)

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
                format(num_g)
        )

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in tqdm(range(num_q)):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc = 0.
        for repeat_idx in range(num_repeats):
            mask = np.zeros(len(raw_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_raw_cmc = raw_cmc[mask]
            _cmc = masked_raw_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)

        cmc /= num_repeats
        all_cmc.append(cmc)
        # compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_market1501(distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank, use_distmat):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    # num_q, num_g = distmat.shape
    num_q = q_pids.shape[0]
    num_g = g_pids.shape[0]

    print("Num queries: ", num_q)
    print("Num gallery_images: ", num_g)

    dim = q_feats.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(g_feats)

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    startx = time()
    if use_distmat:
        indices = np.argsort(distmat, axis=1)
    else:
        _, indices = index.search(q_feats, k=num_g)

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    print("Time to take get matches: ", time() - startx)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in tqdm(range(num_q)):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP


def eval_market1501_parallel(distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank, use_distmat, num_workers=1):
    """Parallel Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    # num_q, num_g = distmat.shape
    num_q = q_pids.shape[0]
    num_g = g_pids.shape[0]

    print("Num queries: ", num_q)
    print("Num gallery_images: ", num_g)

    dim = q_feats.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(g_feats)

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    startx = time()
    if use_distmat:
        indices = np.argsort(distmat, axis=1)
    else:
        _, indices = index.search(q_feats, k=num_g)

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    print("Time to take get matches: ", time() - startx)

    from multiprocessing import Process
    import math
    threads = []
    num_queries_per_thread = math.ceil(num_q / num_workers)
    results = {}

    for worker_idx in range(num_workers):
        start = worker_idx * num_queries_per_thread
        end = min((worker_idx + 1) * num_queries_per_thread, num_q)
        # No need of locks/mutexes as threads aren't modifying common variables.
        # But need to avoid creating new variables for each argument
        # threads.append(threading.Thread(target=compute_stats, args=(start, end, q_pids, q_camids, g_pids, g_camids, indices, matches, max_rank, worker_idx, results)))
        threads.append(Process(target=compute_stats, args=(start, end, q_pids, q_camids, g_pids, g_camids, indices, matches, max_rank, worker_idx, results)))

    # Start all threads
    for worker_idx in range(num_workers):
        print("Starting thread worker ", worker_idx)
        threads[worker_idx].start()

    # Wait for all threads to finish
    for worker_idx in range(num_workers):
        threads[worker_idx].join()

    # Aggregate results
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    temp_dir = "./temp_parallel_results"
    assert os.path.isdir(temp_dir)

    for worker_idx in range(num_workers):
        f = open(os.path.join(temp_dir, f"results_{worker_idx}.pkl"), "rb")
        results = pkl.load(f)
        f.close()
        all_cmc += results[worker_idx]["all_cmc"]
        all_AP += results[worker_idx]["all_AP"]
        all_INP += results[worker_idx]["all_INP"]
        num_valid_q += results[worker_idx]["num_valid_q"]

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    # Delete the temp files and folders
    print("Deleting all temporary folders and files")
    shutil.rmtree(temp_dir, ignore_errors=True)

    return all_cmc, all_AP, all_INP



def compute_stats(start, stop, q_pids, q_camids, g_pids, g_camids, indices, matches, max_rank, worker_idx, results):
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in tqdm(range(start, stop)):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    results[worker_idx] = {}
    results[worker_idx]["all_cmc"] = all_cmc
    results[worker_idx]["all_AP"] = all_AP
    results[worker_idx]["all_INP"] = all_INP
    results[worker_idx]["num_valid_q"] = num_valid_q

    temp_dir = "./temp_parallel_results"
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)

    result_save_file = os.path.join(temp_dir, f"results_{worker_idx}.pkl")
    f = open(result_save_file, 'wb')
    pkl.dump(results, f)
    f.close() 
    return


def evaluate_py(
        distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03, use_distmat, parallelize=False, num_workers=1
):
    if use_metric_cuhk03:
        return eval_cuhk03(
            distmat, q_feats, g_feats, g_pids, q_camids, g_camids, max_rank, use_distmat
        )
    else:
        if not parallelize:
            return eval_market1501(
                distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank, use_distmat
            )
        else:
            assert num_workers > 1, 'Error: Set num_workers greater than 1 for parallelization'
            return eval_market1501_parallel(
                distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank, use_distmat, num_workers
            )


def evaluate_rank(
        distmat,
        q_feats,
        g_feats,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        max_rank=50,
        use_metric_cuhk03=False,
        use_distmat=False,
        use_cython=True,
        parallelize=False,
        num_workers=1
):
    """Evaluates CMC rank.
    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_feats (numpy.ndarray): 2-D array containing query features.
        g_feats (numpy.ndarray): 2-D array containing gallery features.
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
        use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
            Default is False. This should be enabled when using cuhk03 classic split.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """
    if use_cython and IS_CYTHON_AVAI:
        return evaluate_cy(
            distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank,
            use_metric_cuhk03, use_distmat
        )
    else:
        return evaluate_py(
            distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank,
            use_metric_cuhk03, use_distmat, parallelize=parallelize, num_workers=num_workers
        )
