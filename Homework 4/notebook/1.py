# knn_effect_of_k_loocv_fixed5fold.py
import math
import numpy as np
import pandas as pd
from collections import Counter

# ----------------------
# Dataset
# ----------------------
data = np.array([
    [1, 1.0, 6.0, 0],
    [2, 2.2, 7.0, 0],
    [3, 3.1, 8.2, 0],
    [4, 4.0, 6.1, 0],
    [5, 5.2, 6.0, 1],
    [6, 6.0, 6.2, 0],
    [7, 6.2, 4.8, 1],
    [8, 7.0, 3.8, 1],
    [9, 8.2, 4.6, 1],
    [10,8.8, 6.0, 0],
    [11,3.2, 5.4, 1],
    [12,2.8, 6.2, 0],
    [13,7.6, 7.6, 1],
    [14,4.8, 7.4, 1]
])
cols = ['idx','x','y','label']
df = pd.DataFrame(data, columns=cols).astype({'idx':int,'x':float,'y':float,'label':int})
X = df[['x','y']].to_numpy()
y = df['label'].to_numpy().astype(int)

# ----------------------
# Distances
# ----------------------
def d1(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
def d2(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
def dinf(a,b): return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

metrics = {'manhattan': d1, 'euclidean': d2, 'chebyshev': dinf}

# ----------------------
# Neighbor selection in LOOCV context (and general helper)
# - For k==1: include ALL points with minimal distance (to respect equal-distance neighbor voting)
# - For k>1: take top-k by (distance, index) deterministic ordering
# ----------------------
def get_neighbors_for_test(test_pos, X, k, metric_fn, tol=1e-9):
    dists = []
    for j in range(len(X)):
        if j == test_pos:
            continue
        d = metric_fn(X[test_pos], X[j])
        dists.append((j, d))
    dists.sort(key=lambda t: (t[1], t[0]))  # sort by distance then index for determinism
    if k == 1:
        # include all with minimal distance
        min_d = dists[0][1]
        tied = [t for t in dists if math.isclose(t[1], min_d, rel_tol=tol, abs_tol=tol)]
        return tied  # list of (pos, dist)
    else:
        # for k>1, simply take top-k entries (deterministic by distance then index)
        return dists[:k]

# ----------------------
# Prediction from neighbor list: vote and tie-break favoring class 0
# ----------------------
def vote_from_neighbors(neigh_positions, y):
    labels = [int(y[p]) for p in neigh_positions]
    counts = Counter(labels)
    # tie -> class 0
    if counts[0] == counts[1]:
        pred = 0
    else:
        pred = 0 if counts[0] > counts[1] else 1
    return pred, counts, labels

# ----------------------
# LOOCV for given k and metric (tie-aware)
# returns error, misclassified_indices, details (dict)
# details: for each mis idx -> {'neighbors': [(idx,dist,label),...], 'votes': Counter, 'pred':..., 'true':...}
# ----------------------
def loocv_tie_aware(X, y, k, metric_fn):
    n = len(X)
    mis = []
    details = {}
    for i in range(n):
        neigh = get_neighbors_for_test(i, X, k, metric_fn)
        neigh_positions = [p for (p,d) in neigh]
        neigh_dists = [d for (p,d) in neigh]
        pred, counts, labels = vote_from_neighbors(neigh_positions, y)
        true = int(y[i])
        if pred != true:
            mis.append(int(df.iloc[i]['idx']))
            details[int(df.iloc[i]['idx'])] = {
                'neighbors': [(int(df.iloc[p]['idx']), round(d,6), int(df.iloc[p]['label'])) for (p,d) in neigh],
                'votes': counts,
                'pred': pred,
                'true': true
            }
    error = len(mis) / n
    return error, mis, details

# ----------------------
# (b) Compute LOOCV for k in {1,3,5} for each metric; report results
# ----------------------
def compute_all_loocv():
    ks = [1,3,5]
    results = {}
    for mname, mfn in metrics.items():
        results[mname] = {}
        for k in ks:
            err, mis, det = loocv_tie_aware(X, y, k, mfn)
            results[mname][k] = {'error': err, 'mis': mis, 'details': det}
    return results

# ----------------------
# Find best LOOCV setting(s) and choose preferred metric if tie
# ----------------------
def find_best_settings(results):
    best = []
    best_err = min(results[m][k]['error'] for m in results for k in results[m])
    for m in results:
        for k in results[m]:
            if abs(results[m][k]['error'] - best_err) < 1e-12:
                best.append((m,k,results[m][k]['error']))
    # if multiple, prefer euclidean if present
    chosen = None
    for s in best:
        if s[0] == 'euclidean':
            chosen = s
            break
    if chosen is None:
        chosen = best[0]
    return best_err, best, chosen

# ----------------------
# For chosen metric, produce full 5-NN calculation for every misclassified point
# ----------------------
def full_5nn_details_for_metric(metric_name):
    metric_fn = metrics[metric_name]
    k = 5
    # We'll select top-5 by (distance,index). (If k==1 tie-handling already handled elsewhere.)
    n = len(X)
    mis_details = {}
    for i in range(n):
        # compute distances to all other points (exclude self)
        dists = []
        for j in range(n):
            if j == i: continue
            dists.append((j, metric_fn(X[i], X[j])))
        # sort and pick top-5 deterministically
        dists.sort(key=lambda t:(t[1], t[0]))
        topk = dists[:k]
        neigh_positions = [p for (p,d) in topk]
        pred, counts, labels = vote_from_neighbors(neigh_positions, y)
        true = int(y[i])
        if pred != true:
            mis_details[int(df.iloc[i]['idx'])] = {
                'neighbors': [(int(df.iloc[p]['idx']), round(d,6), int(df.iloc[p]['label'])) for (p,d) in topk],
                'votes': counts,
                'pred': pred,
                'true': true
            }
    return mis_details

# ----------------------
# NEW: Evaluate fixed folds (no reshuffle) for a given metric function
# folds: list of lists of idx (1-based index values as in df['idx'])
# For each fold: compute error (num_mis / fold_size) and list misclassified indices in that fold
# ----------------------
def evaluate_fixed_folds(folds, X, y, k, metric_fn):
    n = len(X)
    fold_errors = []
    fold_mis = []
    fold_details = []
    # convert df idx to positions mapping
    idx_to_pos = {int(df.iloc[i]['idx']): i for i in range(len(df))}
    for fold in folds:
        # get test positions (0-based)
        test_positions = [idx_to_pos[idx] for idx in fold]
        train_positions = [p for p in range(n) if p not in test_positions]
        mis_here = []
        details_here = {}
        for pos in test_positions:
            # compute distances from test point to all training points
            dists = []
            for jt in train_positions:
                d = metric_fn(X[pos], X[jt])
                dists.append((jt, d))
            dists.sort(key=lambda t:(t[1], t[0]))
            if k == 1:
                # include all tied nearest neighbors from training set
                if not dists:
                    neigh = []
                else:
                    min_d = dists[0][1]
                    neigh = [t for t in dists if math.isclose(t[1], min_d, rel_tol=1e-9, abs_tol=1e-9)]
            else:
                neigh = dists[:k]
            neigh_positions = [p for (p,d) in neigh]
            pred, counts, labels = vote_from_neighbors(neigh_positions, y)
            true = int(df.iloc[pos]['label'])
            if pred != true:
                mis_here.append(int(df.iloc[pos]['idx']))
                details_here[int(df.iloc[pos]['idx'])] = {
                    'neighbors': [(int(df.iloc[p]['idx']), round(d,6), int(df.iloc[p]['label'])) for (p,d) in neigh],
                    'votes': counts,
                    'pred': pred,
                    'true': true
                }
        fold_errors.append(len(mis_here)/len(test_positions))
        fold_mis.append(mis_here)
        fold_details.append(details_here)
    return fold_errors, fold_mis, fold_details

# ----------------------
# Run everything and print human readable report
# ----------------------
if __name__ == "__main__":
    # LOOCV summary (tie-aware)
    results = compute_all_loocv()

    print("LOOCV results (tie-aware) for each metric and k in {1,3,5}:\n")
    for m in sorted(results.keys()):
        for k in sorted(results[m].keys()):
            info = results[m][k]
            print(f" Metric={m:9s}  k={k:1d}  LOOCV_error = {info['error']:.6f}  misclassified = {info['mis']}")
    print("\n" + "="*70 + "\n")

    best_err, best_list, chosen = find_best_settings(results)
    print(f"Best LOOCV error = {best_err:.6f}")
    print("All best settings (metric, k, error):", best_list)
    print("Chosen setting for detailed 5-NN output (prefer euclidean if tie):", chosen)
    print("\n" + "="*70 + "\n")

    # For the chosen metric, compute full 5-NN details for every misclassified point (neighbor IDs, labels, distances, vote)
    chosen_metric = chosen[0]
    print(f"Full 5-NN details for metric = {chosen_metric} (using k=5) — showing every misclassified point:\n")
    mis5 = full_5nn_details_for_metric(chosen_metric)
    if not mis5:
        print(" No misclassified points under 5-NN for this metric.")
    else:
        for idx, info in sorted(mis5.items()):
            print(f"Test idx {idx}: true={info['true']}, pred={info['pred']}")
            print("  5-NN (ID, distance, label):")
            for nb in info['neighbors']:
                print(f"    {nb}")
            print("  votes:", dict(info['votes']))
            print("-"*40)

    # ----------------------
    # NEW: Fixed 5-fold CV (Euclidean) evaluation (no reshuffle)
    # ----------------------
    folds = [
        [1,7,11],   # Fold1
        [2,8,12],   # Fold2
        [3,9,5],    # Fold3
        [4,6,10],   # Fold4
        [13,14]     # Fold5
    ]

    print("\n" + "="*70 + "\n")
    print("Fixed 5-fold CV results (Euclidean d2), k in {1,3,5}, no reshuffle:\n")

    fold_results_all_k = {}
    for k in [1,3,5]:
        errs, missets, details = evaluate_fixed_folds(folds, X, y, k, metrics['euclidean'])
        fold_results_all_k[k] = {'fold_errors': errs, 'fold_mis': missets, 'fold_details': details, 'mean_error': sum(errs)/len(errs)}
        print(f" k = {k}")
        for fi, fold in enumerate(folds, start=1):
            print(f"  Fold {fi} (test indices {fold}): error = {errs[fi-1]:.6f}, misclassified = {missets[fi-1]}")
        print(f"  mean error across folds = {fold_results_all_k[k]['mean_error']:.6f}")
        print("-"*40)

    # choose recommended k by this fixed split (min mean error). If tie, choose smallest k (or change tie-break as desired)
    best_k = min(fold_results_all_k.keys(), key=lambda kk: (fold_results_all_k[kk]['mean_error'], kk))
    print("\nRecommendation from this fixed 5-fold split:")
    print(f" Recommended k = {best_k} with mean error = {fold_results_all_k[best_k]['mean_error']:.6f}")
    # brief justification
    print(" Justification: selected k minimizes the mean fold error on the provided fixed split.")
    print("\nDetailed misclassifications per fold for recommended k:")
    for fi, fold in enumerate(folds, start=1):
        print(f"  Fold {fi} test indices {fold} -> misclassified = {fold_results_all_k[best_k]['fold_mis'][fi-1]}")