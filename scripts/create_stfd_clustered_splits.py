#!/usr/bin/env python3
"""
TRUSTTRACE: Candidate Base-Template Clustered Dataset Splitting Script.

Repairs train/val/test splits using candidate template clustering:
- Constructs an undirected similarity graph with edge condition: pHash <= 4
- Extracts connected components as candidate template clusters
- Each cluster is assigned in its entirety to either TRAIN, VAL, or TEST (never split)
- Two-stage deterministic greedy group-stratification (seed 42) ensures balanced
  manipulation class proportions while strictly maintaining cluster isolation
- Outputs:
  1. data/manifests/trusttrace_stfd_clustered_master.csv
  2. data/manifests/trusttrace_stfd_clustered_train.csv
  3. data/manifests/trusttrace_stfd_clustered_val.csv
  4. data/manifests/trusttrace_stfd_clustered_test.csv
  5. data/manifests/trusttrace_stfd_candidate_edges.csv

100% read-only access to STFD images.
"""

import sys
import os
import csv
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = WORKSPACE_DIR / "data" / "manifests"

MASTER_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_master.csv"
CLUSTERED_MASTER_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_master.csv"
CLUSTERED_TRAIN_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_train.csv"
CLUSTERED_VAL_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_val.csv"
CLUSTERED_TEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_test.csv"
CANDIDATE_EDGES_PATH = MANIFESTS_DIR / "trusttrace_stfd_candidate_edges.csv"

RANDOM_SEED = 42
TARGET_WEIGHTS = {"TRAIN": 0.70, "VAL": 0.15, "TEST": 0.15}
PHASH_EDGE_THRESHOLD = 4
CLUSTER_ASSIGNMENT_METHOD = "PHASH_LE4_CONNECTED_COMPONENTS"


def compute_phash_64(img_gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """
    Computes a 64-bit DCT perceptual hash (pHash) as a boolean array of shape (64,).
    """
    img_size = hash_size * highfreq_factor
    resized = cv2.resize(img_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_lowfreq = dct[:hash_size, :hash_size]
    med = np.median(dct_lowfreq[1:, 1:])
    return (dct_lowfreq > med).flatten()


def compute_dhash_64(img_gray: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """
    Computes a 64-bit difference hash (dHash) as a boolean array of shape (64,).
    """
    resized = cv2.resize(img_gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()


def load_master_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Master manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != 3932:
        raise ValueError(f"Expected exactly 3,932 rows in master manifest, found {len(rows)}")
    return rows


def compute_hashes(records: List[Dict[str, Any]], num_workers: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    n = len(records)
    phash_bits = np.zeros((n, 64), dtype=np.uint8)
    dhash_bits = np.zeros((n, 64), dtype=np.uint8)

    def _worker(idx):
        p = records[idx]["image_path"]
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        return idx, compute_phash_64(im), compute_dhash_64(im)

    print(f"Computing pHash and dHash for {n} images across {num_workers} threads...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for idx, ph, dh in executor.map(_worker, range(n)):
            phash_bits[idx] = ph
            dhash_bits[idx] = dh
    t_elapsed = time.time() - t0
    print(f"Hash computation completed in {t_elapsed:.2f}s.", flush=True)
    return phash_bits, dhash_bits


def compute_pairwise_distances(bits: np.ndarray) -> np.ndarray:
    sum_bits = bits.sum(axis=1, dtype=np.int32)
    dot = bits.astype(np.int32) @ bits.T.astype(np.int32)
    return sum_bits[:, None] + sum_bits[None, :] - 2 * dot


def build_candidate_clusters(
    records: List[Dict[str, Any]],
    p_dists: np.ndarray,
    d_dists: np.ndarray,
    edge_threshold: int = PHASH_EDGE_THRESHOLD
) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    n = len(records)
    adj = defaultdict(list)
    edges = []

    print(f"Building candidate similarity graph with pHash Hamming distance <= {edge_threshold}...", flush=True)
    for i in range(n):
        for j in range(i + 1, n):
            pd = int(p_dists[i, j])
            if pd <= edge_threshold:
                dd = int(d_dists[i, j])
                adj[i].append(j)
                adj[j].append(i)
                edges.append({
                    "sample_id_A": records[i]["sample_id"],
                    "sample_id_B": records[j]["sample_id"],
                    "pHash_distance": pd,
                    "dHash_distance": dd,
                    "manipulation_A": records[i]["manipulation_type"],
                    "manipulation_B": records[j]["manipulation_type"],
                })

    print(f"Graph constructed: {n} nodes, {len(edges)} candidate edges.", flush=True)

    # Connected components using deterministic traversal
    visited = set()
    raw_clusters = []
    for i in range(n):
        if i not in visited:
            comp = []
            stack = [i]
            visited.add(i)
            while stack:
                curr = stack.pop()
                comp.append(curr)
                for nb in adj[curr]:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            raw_clusters.append(comp)

    # Sort each cluster deterministically by sample_id
    clusters = []
    for comp in raw_clusters:
        sorted_comp = sorted(comp, key=lambda idx: records[idx]["sample_id"])
        clusters.append(sorted_comp)

    # Sort clusters deterministically by the first sample_id
    clusters.sort(key=lambda c: records[c[0]]["sample_id"])

    return clusters, edges


def assign_clusters_to_splits(
    records: List[Dict[str, Any]],
    clusters: List[List[int]],
    seed: int = RANDOM_SEED
) -> Dict[str, List[List[int]]]:
    """
    Two-stage greedy group-stratification algorithm:
    Stage 1: Multi-image clusters are assigned to TRAIN/VAL/TEST while balancing capacity and class need.
    Stage 2: Singleton clusters are assigned to balance exact manipulation class deficits.
    Every cluster remains 100% indivisible.
    """
    n = len(records)
    splits = ["TRAIN", "VAL", "TEST"]
    manip_types = sorted(list(set(r["manipulation_type"] for r in records)))
    total_manip = {mt: sum(1 for r in records if r["manipulation_type"] == mt) for mt in manip_types}

    target_tot = {s: int(round(n * TARGET_WEIGHTS[s])) for s in splits}
    diff = n - sum(target_tot.values())
    target_tot["TRAIN"] += diff  # 2752, 590, 590

    target_manip = {
        s: {mt: total_manip[mt] * TARGET_WEIGHTS[s] for mt in manip_types}
        for s in splits
    }

    rng = np.random.RandomState(seed)

    multi_clusters = [c for c in clusters if len(c) > 1]
    singleton_clusters = [c for c in clusters if len(c) == 1]

    # Stage 1: Multi-image clusters
    # Sort descending by size, permute ties deterministically
    by_size = defaultdict(list)
    for c in multi_clusters:
        by_size[len(c)].append(c)

    ordered_multi = []
    for sz in sorted(by_size.keys(), reverse=True):
        grp = by_size[sz]
        perm = rng.permutation(len(grp))
        for p_idx in perm:
            ordered_multi.append(grp[p_idx])

    current_splits = {s: [] for s in splits}
    current_manip = {s: defaultdict(int) for s in splits}
    current_total = {s: 0 for s in splits}

    for c in ordered_multi:
        c_size = len(c)
        c_manip = defaultdict(int)
        for idx in c:
            c_manip[records[idx]["manipulation_type"]] += 1

        best_split = None
        best_score = -float("inf")

        for s in splits:
            cap_room = target_tot[s] - (current_total[s] + c_size)
            manip_need = 0.0
            for mt, cnt in c_manip.items():
                manip_need += (target_manip[s][mt] - current_manip[s][mt])

            score = cap_room / target_tot[s] + 1.5 * (manip_need / sum(target_manip[s].values()))
            if score > best_score:
                best_score = score
                best_split = s

        current_splits[best_split].append(c)
        current_total[best_split] += c_size
        for mt, cnt in c_manip.items():
            current_manip[best_split][mt] += cnt

    # Stage 2: Singleton clusters
    perm_singles = rng.permutation(len(singleton_clusters))
    shuffled_singles = [singleton_clusters[i] for i in perm_singles]

    for c in shuffled_singles:
        idx = c[0]
        mt = records[idx]["manipulation_type"]

        best_split = None
        best_need = -float("inf")

        for s in splits:
            cap_room = target_tot[s] - current_total[s]
            m_deficit = target_manip[s][mt] - current_manip[s][mt]
            need = m_deficit + 0.1 * cap_room
            if cap_room <= 0:
                need -= 1000.0

            if need > best_need:
                best_need = need
                best_split = s

        current_splits[best_split].append(c)
        current_total[best_split] += 1
        current_manip[best_split][mt] += 1

    return current_splits


def main():
    print("=" * 70)
    print("TRUSTTRACE: CANDIDATE TEMPLATE CLUSTERED DATASET GENERATOR")
    print("=" * 70)

    # 1. Load master manifest
    records = load_master_manifest(MASTER_MANIFEST_PATH)
    n = len(records)
    print(f"Loaded {n} records from {MASTER_MANIFEST_PATH.name}.")

    # 2. Compute fingerprints
    phash_bits, dhash_bits = compute_hashes(records)

    # 3. Compute pairwise distances
    print("Computing pairwise distance matrices...", flush=True)
    p_dists = compute_pairwise_distances(phash_bits)
    d_dists = compute_pairwise_distances(dhash_bits)

    # 4. Build clusters and edges
    clusters, edges = build_candidate_clusters(records, p_dists, d_dists, PHASH_EDGE_THRESHOLD)
    num_clusters = len(clusters)
    cluster_sizes = [len(c) for c in clusters]
    singletons = sum(1 for s in cluster_sizes if s == 1)
    multi_clusters = num_clusters - singletons

    print(f"\nCluster Overview:")
    print(f"  Total Candidate Template Clusters: {num_clusters}")
    print(f"  Singleton Clusters (size = 1):     {singletons} ({singletons / n * 100:.2f}% of images)")
    print(f"  Multi-image Clusters (size > 1):   {multi_clusters} ({sum(s for s in cluster_sizes if s > 1)} images)")
    print(f"  Cluster Size Min / Median / Max:   {min(cluster_sizes)} / {np.median(cluster_sizes):.1f} / {max(cluster_sizes)}")

    # Assign cluster IDs (cluster_0001 to cluster_XXXX)
    cluster_id_map = {}
    sample_to_cluster_id = {}
    for c_idx, c in enumerate(clusters, start=1):
        c_id = f"cluster_{c_idx:04d}"
        cluster_id_map[c_idx] = c_id
        for sample_idx in c:
            sample_to_cluster_id[records[sample_idx]["sample_id"]] = c_id

    # 5. Assign clusters to splits
    print("\nAssigning clusters to TRAIN / VAL / TEST partitions...", flush=True)
    split_clusters = assign_clusters_to_splits(records, clusters, seed=RANDOM_SEED)

    # Map samples to splits
    sample_to_split = {}
    cluster_to_split = {}
    split_samples = defaultdict(list)

    for s, c_list in split_clusters.items():
        for c in c_list:
            c_id = sample_to_cluster_id[records[c[0]]["sample_id"]]
            cluster_to_split[c_id] = s
            for sample_idx in c:
                sid = records[sample_idx]["sample_id"]
                sample_to_split[sid] = s
                split_samples[s].append(records[sample_idx])

    # 6. Verify cluster integrity across splits
    print("\n--- CLUSTER ISOLATION VERIFICATION ---")
    train_clusters = set(sample_to_cluster_id[r["sample_id"]] for r in split_samples["TRAIN"])
    val_clusters = set(sample_to_cluster_id[r["sample_id"]] for r in split_samples["VAL"])
    test_clusters = set(sample_to_cluster_id[r["sample_id"]] for r in split_samples["TEST"])

    print(f"TRAIN Clusters: {len(train_clusters)}")
    print(f"VAL Clusters:   {len(val_clusters)}")
    print(f"TEST Clusters:  {len(test_clusters)}")

    leak_tr_val = train_clusters & val_clusters
    leak_tr_test = train_clusters & test_clusters
    leak_val_test = val_clusters & test_clusters

    if leak_tr_val or leak_tr_test or leak_val_test:
        raise RuntimeError(f"FATAL: Cluster leakage detected! TR-VAL: {leak_tr_val}, TR-TE: {leak_tr_test}, VAL-TE: {leak_val_test}")
    print("[PASS] ZERO cluster overlap across any split boundaries!")

    # 7. Verify cross-split perceptual isolation
    print("\n--- CROSS-SPLIT PERCEPTUAL AUDIT ---")
    cross_0 = 0
    cross_4 = 0
    cross_8 = 0
    for i in range(n):
        for j in range(i + 1, n):
            s_i = sample_to_split[records[i]["sample_id"]]
            s_j = sample_to_split[records[j]["sample_id"]]
            if s_i != s_j:
                pd = p_dists[i, j]
                if pd == 0: cross_0 += 1
                if pd <= 4: cross_4 += 1
                if pd <= 8: cross_8 += 1

    print(f"  Cross-Split pHash == 0: {cross_0} (Expected: 0)")
    print(f"  Cross-Split pHash <= 4: {cross_4} (Expected: 0)")
    print(f"  Cross-Split pHash <= 8: {cross_8}")

    if cross_0 != 0 or cross_4 != 0:
        raise RuntimeError(f"FATAL: Perceptual leakage detected! cross_0={cross_0}, cross_4={cross_4}")
    print("[PASS] ZERO cross-split exact duplicates (pHash=0) and ZERO cross-split candidate pairs (pHash<=4)!")

    # 8. Report split sizes and manipulation distributions
    print("\n--- PARTITION SUMMARY ---")
    manip_types = sorted(list(set(r["manipulation_type"] for r in records)))
    for s in ["TRAIN", "VAL", "TEST"]:
        cnt = len(split_samples[s])
        print(f"  {s:<5}: {cnt:>4} samples ({cnt / n * 100:.2f}%)")

    print("\n--- MANIPULATION CLASS DISTRIBUTION ---")
    for mt in manip_types:
        tot = sum(1 for r in records if r["manipulation_type"] == mt)
        tr = sum(1 for r in split_samples["TRAIN"] if r["manipulation_type"] == mt)
        va = sum(1 for r in split_samples["VAL"] if r["manipulation_type"] == mt)
        te = sum(1 for r in split_samples["TEST"] if r["manipulation_type"] == mt)
        print(f"  {mt:<16} | Train: {tr:>4} ({tr/tot*100:.1f}%) | Val: {va:>4} ({va/tot*100:.1f}%) | Test: {te:>4} ({te/tot*100:.1f}%) | Total: {tot}")

    # 9. Write candidate edges file
    print(f"\nWriting candidate edge list to {CANDIDATE_EDGES_PATH.name}...", flush=True)
    with open(CANDIDATE_EDGES_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id_A", "sample_id_B", "pHash_distance", "dHash_distance", "manipulation_A", "manipulation_B"
        ])
        writer.writeheader()
        writer.writerows(edges)
    print(f"Saved {len(edges)} candidate edges.", flush=True)

    # 10. Write clustered manifests
    # Fieldnames include candidate_template_cluster_id and cluster_assignment_method
    fieldnames = list(records[0].keys())
    if "candidate_template_cluster_id" not in fieldnames:
        fieldnames.append("candidate_template_cluster_id")
    if "cluster_assignment_method" not in fieldnames:
        fieldnames.append("cluster_assignment_method")

    # Enrich records
    enriched_records = []
    for r in records:
        r_copy = dict(r)
        r_copy["candidate_template_cluster_id"] = sample_to_cluster_id[r["sample_id"]]
        r_copy["cluster_assignment_method"] = CLUSTER_ASSIGNMENT_METHOD
        enriched_records.append(r_copy)

    print(f"Writing clustered master manifest to {CLUSTERED_MASTER_PATH.name}...", flush=True)
    with open(CLUSTERED_MASTER_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_records)

    split_paths = {
        "TRAIN": CLUSTERED_TRAIN_PATH,
        "VAL": CLUSTERED_VAL_PATH,
        "TEST": CLUSTERED_TEST_PATH,
    }

    for s, path in split_paths.items():
        recs = [r for r in enriched_records if sample_to_split[r["sample_id"]] == s]
        # Sort deterministically by sample_id
        recs.sort(key=lambda x: x["sample_id"])
        print(f"Writing {s} manifest ({len(recs)} rows) to {path.name}...", flush=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(recs)

    print("\n" + "=" * 70)
    print("SUCCESS: ALL CLUSTERED SPLITS GENERATED AND VERIFIED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    main()
