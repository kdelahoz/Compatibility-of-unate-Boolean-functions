# -*- coding: utf-8 -*-
"""
Coverage Algorithm (Algorithm 1).
"""

import random
import math
import copy
from scipy.stats import norm

from src.discrepancies import generate_discrepancies

# -------------------------
# Main functions 
# -------------------------

def weight(vector_sign):
    """Compute the weight of a signed vector."""
    count = 0
    for k in range(len(vector_sign)):
        if vector_sign[k] != '0':
            count += 1
    return count

def prune_redundant_coordinates(D, S):
    """
    Redundancy-guided pruning of a coverage vector.

    Removes coordinates whose covered discrepancies remain covered
    by other selected coordinates. The result is inclusion-minimal.
    """
    S = S.copy()
    selected = [i for i, s in enumerate(S) if s != "0"]

    if not selected:
        return S, {"removed_count": 0, "removed_coordinates": []}

    # For each selected coordinate, store the discrepancies it covers.
    covered_by = {
        i: [q for q, delta in enumerate(D) if delta[i] == S[i]]
        for i in selected
    }

    # Count how many selected coordinates cover each discrepancy.
    cover_count = [0] * len(D)
    for qs in covered_by.values():
        for q in qs:
            cover_count[q] += 1

    # Rank coordinates by redundancy: larger minimum backup coverage first.
    redundancy = {
        i: min((cover_count[q] - 1 for q in covered_by[i]), default=float("inf"))
        for i in selected
    }

    removed = []

    for i in sorted(selected, key=lambda i: (-redundancy[i], i)):
        if all(cover_count[q] >= 2 for q in covered_by[i]):
            S[i] = "0"
            removed.append(i)
            for q in covered_by[i]:
                cover_count[q] -= 1

    return S, {
        "removed_count": len(removed),
        "removed_coordinates": removed,
    }


def binomial_prob(nt, p):
    """Sigmoid-like probability based on Normal approximation of Binomial."""
    if p == 1:
        return 1
    elif p == 0:
        return 0  # should not happen though
    else:
        return norm.cdf((p - 1/2) / math.sqrt(p * (1 - p) / nt))


def distribu_indices(liste):
    """Choose an index k from a probability distribution given by 'liste'."""
    z = random.random()
    k = -1
    somme = 0
    while somme < z:
        k += 1
        somme += liste[k]
    return k


def testable_indices(MODshrinking, n):
    """Indexes of columns that have at least one +/- discrepancy remaining."""
    testable = []
    for k in range(n):
        if MODshrinking[0][k] != 0 or MODshrinking[1][k] != 0:
            testable.append(k)
    return testable


def count_potential_discrep_deleted(D, sign, colonne):
    """Count discrepancies that would be covered by assigning 'sign' at 'colonne'."""
    d = len(D)
    count = 0
    for u in range(d):
        if D[u][colonne] == sign:
            count += 1
    return count


def countline(D, l, Snong, n):
    """Size of the support of discrepancy l restricted to unassigned coordinates."""
    count = 0
    for k in range(n):
        if Snong[k] == "0" and D[l][k] != "0":
            count += 1
    return count


def arrangediscrepances(D, deletedlinesNONGLOBAL, Snong, n):
    """Bucket discrepancies by increasing weight among the not-covered ones."""
    d = len(D)
    WW = [[] for _ in range(n + 1)]
    for q in range(d):
        if q not in deletedlinesNONGLOBAL:
            wx = countline(D, q, Snong, n)
            WW[wx].append(q)
    return WW


def list_discrepances_smallest_weight(D, deletedlinesNONGLOBAL, Snong, n):
    """Return discrepancies of smallest positive weight."""
    WW = arrangediscrepances(D, deletedlinesNONGLOBAL, Snong, n)
    prem = "deadend_3"  # no discrepancies left
    sw = "deadend_3"
    if WW[0]:
        return ["deadend_3", 0]
    for k in range(1, n + 1):
        if len(WW[k]) != 0:
            prem = WW[k]
            sw = k
            break
    return [prem, sw]


def unattributed_support(D, q, S, n):
    """Support of discrepancy q intersected with unassigned coordinates in S."""
    support = []
    for u in range(n):
        if D[q][u] != "0" and S[u] == "0":
            support.append(u)
    return support


def sig(D, l, c):
    """Binary version of D[l][c] where '+'=1 and '-'=0."""
    if D[l][c] == "+":
        return 1
    elif D[l][c] == "-":
        return 0
    else:
        return "error"


def how_probable(shrinking_of_list_counts, column, sign):
    """
    How probable assign S[column] = sign.
    shrinking_of_list_counts = [[#minus per col], [#plus per col]]
    """
    countplus = shrinking_of_list_counts[1][column]
    countminus = shrinking_of_list_counts[0][column]
    nt = countminus + countplus
    if countplus < countminus:
        mama = "-"
        p = countminus / nt
    else:
        mama = "+"
        p = countplus / nt
    if (countplus + countminus) == 0:
        return 0
    else:
        bibi = binomial_prob(nt, p) #the most likely ones are the ones that are too close from a binomial: do not choose them
        if mama == sign: #the majority potential is coherent with the sign
            pass
        else:
            bibi = 1 - bibi
    return bibi


def normcoor(D, shrinking_of_list_counts, vector, discrep_chosen, n):
    """Choose a coordinate within the unattributed support (Rule B component)."""
    count = 0
    VVV = []
    for column in range(n):
        if column not in vector:
            VVV.append(0)
        else:
            hhh = how_probable(shrinking_of_list_counts, column, D[discrep_chosen][column])
            VVV.append(hhh)
            count += hhh

    for i in range(len(VVV)):
        if count == 0:
            return "dead_end_5" #the selected discrepancy only has 0s
        else:
            VVV[i] = VVV[i] / count

    ddd = distribu_indices(VVV)
    return ddd


def normdiscrep(D, shrinking_of_list_counts, listdisc, S, n):
    """Choose a discrepancy among smallest-weight ones (Rule A component)."""
    vector = []
    total = 0

    for discrep_index in listdisc:
        count = 0
        support = unattributed_support(D, discrep_index, S, n)
        for k in support:
            count += shrinking_of_list_counts[sig(D, discrep_index, k)][k]
            total += shrinking_of_list_counts[sig(D, discrep_index, k)][k]
        vector.append(count)

    if total == 0:
        return "deadend" #This condition should be impossible because each coordinate covers at least its discrepancy.
    else:
        for vec in range(len(vector)):
            vector[vec] = vector[vec] / total
        dididi = distribu_indices(vector)
        lll = listdisc[dididi]
        return lll


def prob_signs_from_support(D, shrinking_of_list_counts, WWk, S, n):
    """Rule A + Rule B combined selection."""
    list_of_discrep_of_smallest_weight = WWk[0]
    if list_of_discrep_of_smallest_weight == "deadend_3":
        return "deadend_4"

    discrep_chosen = normdiscrep(D, shrinking_of_list_counts, list_of_discrep_of_smallest_weight, S, n)
    if discrep_chosen == "deadend":
        return "deadend_2"
    else:
        una_disc = unattributed_support(D, discrep_chosen, S, n)
        coord_chosen = normcoor(D, shrinking_of_list_counts, una_disc, discrep_chosen, n)
        if coord_chosen == "dead_end_5":
            return "dead_end_6"
        else:
            return [discrep_chosen, coord_chosen]


# -------------------------
# API Algorithm 1
# -------------------------

def coverage_algorithm(F0, F1, n, k_max=None, seed=None, stop_params=None, prune=True):
    """
    Compute a coverage vector Sigma.

    Parameters
    ----------
    F0 : array-like
        Binary observations with output 0.
    F1 : array-like
        Binary observations with output 1.
    n : int
        Number of variables.
    k_max : int or None
        If not None, enforce a max weight constraint on Sigma.
    seed : int or None
        Random seed for reproducibility (controls random choices in Rule A/B).
    stop_params : dict or None
        Parameters for the stopping condition used in the experiments.
          - "factor" (float): multiplicative scale on the default log-bound.
          - "log_max" (float): absolute override of the log-bound.
            If provided, takes precedence over "factor".
    prune : bool
        If True, remove redundant selected coordinates after a coverage vector is found.

    Returns
    -------
    D : list of list of str
        Multiset of discrepancies generated from F0 and F1.
    Sigma : list of str or None
        Coverage vector in {'0','+','-'}^n if found; otherwise None.
    info : dict
        Extra information useful for replication, including the number of
        discrepancies, whether a solution was found, the weight of Sigma, and
        the number of restarts.
    """
    if seed is not None:
        random.seed(seed)

    # Build multiset of discrepancies
    D = generate_discrepancies(n, F0, F1)
    d = len(D)
    
    info = {"d": d, "found": False, "sigma_weight": 0, "restarts": -1}
    
    # Build initial counts
    MODshrinking_of_list_counts = [[], []] #amount + and - by columns
    for coor in range(n):
        MODshrinking_of_list_counts[0].append(count_potential_discrep_deleted(D, "-", coor))
    for coor in range(n):
        MODshrinking_of_list_counts[1].append(count_potential_discrep_deleted(D, "+", coor))
    # Initial parameters
    S = ["0"] * n
    deletedlines = [] #indices of covered discrepancies
    shrinking_of_list_counts = copy.deepcopy(MODshrinking_of_list_counts)
    # Stop condition parameters
    default_log_max = 1.78277 + 1.15688 * math.log(d) + 0.07946 * math.log(n)
    if stop_params is None:
        log_maximal_complexity = default_log_max
    elif "log_max" in stop_params:
        log_maximal_complexity = stop_params["log_max"]
        if "factor" in stop_params:
            print("[WARN] stop_params has both 'log_max' and 'factor'; "
              "'log_max' takes precedence.")
    else:
        factor = stop_params.get("factor", 1.0)
        log_maximal_complexity = default_log_max * factor
        
    compteuroperations = 1
    testable = testable_indices(shrinking_of_list_counts, n)  #non constant candidate variables
    lentes = len(testable) #number of candidatates
    while len(list(set(deletedlines))) != d:
        if math.log(compteuroperations) > log_maximal_complexity:
            return D, None, info
        # restart
        info["restarts"] += 1
        S = ["0"] * n
        deletedlines = []
        temp_deletedlines = []
        triedindex = []
        shrinking_of_list_counts = copy.deepcopy(MODshrinking_of_list_counts)
        dea = 0
        while len(list(set(temp_deletedlines))) != d and len(set(triedindex)) != lentes and math.log(compteuroperations) < log_maximal_complexity:
            WWk = list_discrepances_smallest_weight(D, temp_deletedlines, S, n)
            dea = prob_signs_from_support(D, shrinking_of_list_counts, WWk, S, n)
            if dea == "deadend_2" or dea == "deadend_4" or dea == "dead_end_6":
                #deadend_2 if WWk[0] do not cover any discrepancy, equal deadend
                #deadend_4 if WWk[0]=[], equal deadend_3
                #dead_end_6 if the selected discrepancy only has 0s, equal dead_end_5                
                compteuroperations += 1
                break
            
            else:
                [s, t] = dea #([discrep_chosen, coord_chosen])
                S[t] = D[s][t]
                if k_max is not None and weight(S) > k_max:
                        break
                triedindex.append(t)
                for f in range(d):
                    compteuroperations += 1
                    if f not in temp_deletedlines:
                        if D[f][t] == S[t]:
                            temp_deletedlines.append(f)
                            for j in range(n): #updates the number of + and - per column
                                if D[f][j] == "-" and S[j] == "0":
                                    shrinking_of_list_counts[0][j] -= 1
                                elif D[f][j] == "+" and S[j] == "0":
                                    shrinking_of_list_counts[1][j] -= 1
                            shrinking_of_list_counts[0][t] = 0 #make the assigned coordinate nontestable
                            shrinking_of_list_counts[1][t] = 0
                temp_deletedlines = list(set(temp_deletedlines))
                triedindex = list(set(triedindex))
        deletedlines = list(set(temp_deletedlines))
    if len(set(deletedlines)) == d:
        info["found"] = True
        info["sigma_weight_before_pruning"] = weight(S)
        
        if prune:
            S, prune_info = prune_redundant_coordinates(D, S)
            info.update(prune_info)
        else:
            info["removed_count"] = 0
            info["removed_coordinates"] = []

        info["sigma_weight"] = weight(S)
        return D, S, info
    return D, None, info
