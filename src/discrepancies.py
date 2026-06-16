# -*- coding: utf-8 -*-
"""
Discrepancy-related functions.
"""

# -------------------------
# Main functions
# -------------------------

def discrepance(x, y, n):
    """Compute the discrepancy between two binary vectors x and y."""
    dis = []
    for k in range(n):
        if x[k] < y[k]:
            dis.append("+")
        elif x[k] > y[k]:
            dis.append("-")
        else:
            dis.append("0")
    return dis


def generate_discrepancies(n, F0, F1):
    """Generate the multiset of discrepancies from the observations."""
    multiset = []
    for x in F0:
        for y in F1:
            dis = discrepance(x, y, n)
            multiset.append(dis)
    return multiset
