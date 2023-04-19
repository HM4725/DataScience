#!/usr/bin/env python3

from __future__ import annotations
import pandas as pd
import numpy as np
import math


class AprioriTable:
    itemsets: np.ndarray[np.ndarray[int]]
    supports: np.ndarray[int]
    level: int

    def __init__(self, itemsets: np.ndarray[np.ndarray[int]], supports: np.ndarray[int] = None):
        self.itemsets = itemsets
        self.supports = supports
        self.level = itemsets[0].sum() if len(self.itemsets) > 0 else 0

    def scan(self, tdb: np.ndarray[np.ndarray[int]]) -> AprioriTable:
        self.supports = (self.itemsets @ tdb.T == self.level).sum(axis=1)
        return self

    def frequents(self, min_sup: int = 0) -> AprioriTable:
        index = np.where(self.supports >= min_sup)
        return AprioriTable(self.itemsets[index], self.supports[index])

    def infrequents(self, min_sup: int = 0) -> AprioriTable:
        index = np.where(self.supports < min_sup)
        return AprioriTable(self.itemsets[index], self.supports[index])

    def join(self) -> AprioriTable:
        n = len(self.itemsets)
        if n > 0:
            joined = self.itemsets.repeat(
                n, 0) | np.tile(self.itemsets, (n, 1))
            indices = np.where(joined.sum(axis=1) == self.level + 1)[0]
            return AprioriTable(np.unique(joined[indices], axis=0))
        else:
            return AprioriTable([])

    def prune(self, infrequents: AprioriTable) -> AprioriTable:
        if len(infrequents) > 0:
            indices = (self.itemsets @ infrequents.itemsets.T <
                       infrequents.level).all(axis=1)
            return AprioriTable(self.itemsets[indices], self.supports)
        else:
            return AprioriTable(self.itemsets, self.supports)
        
    def __iter__(self):
        return zip(self.itemsets, self.supports)

    def __len__(self):
        return len(self.itemsets)
    
def procedure(tdb: np.ndarray[np.ndarray[int]], min_sup: int):
    num_items = tdb.shape[1]
    itemsets = np.diag(np.ones(num_items, dtype=np.int64))
    C = AprioriTable(itemsets).scan(tdb)
    L = C.frequents(min_sup)
    for itemset, support in iter(L):
        yield list(np.where(np.array(itemset) == 1)[0]), support
    while len(L) > 0:
        C = L.join().prune(C.infrequents(min_sup)).scan(tdb)
        L = C.frequents(min_sup)
        for itemset, support in iter(L):
            yield list(np.where(np.array(itemset) == 1)[0]), support


def mine(df: pd.DataFrame, min_support: float) -> pd.DataFrame:
    items = df.columns
    tdb = df.values.astype(np.int64)
    min_sup = math.ceil(min_support * len(tdb))

    itemsets = []
    supports = []
    for itemset, support  in procedure(tdb, min_sup):
        itemsets.append(itemset)
        supports.append(support / len(tdb))
    res_df = pd.DataFrame({"itemset": itemsets, "support": supports})
    res_df["itemset"] = res_df["itemset"].apply(
        lambda x: sorted([items[i] for i in x])
    )
    return res_df
