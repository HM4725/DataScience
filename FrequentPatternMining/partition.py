#!/usr/bin/env python3

from __future__ import annotations
import sys
from functools import reduce
from utils.dataloader import DataLoader
from collections import defaultdict

try:
    from tqdm import tqdm
except:
    def tqdm(x): return x


class DataMiner:
    """Data miner to use the apriori alogirthm with set

    Methods
    ------
    prepare() -> itemsets:
        Apriori algorithm
    mining(itemset) -> patterns:
        Get association rules (patterns)
    log(patterns):
        Log patterns
    """
    dataloader: DataLoader
    sup_min_prob: float
    ntxns: int
    sup_min: int
    k: int

    class AprioriTable:
        itemsets: list[set]
        supports: list[int]

        def __init__(self, itemsets: list[set], supports: list[int] = []):
            self.itemsets = list(itemsets)
            self.supports = list(supports)

        def scan(self, txns: list[list[int]]) -> DataMiner.AprioriTable:
            supports = [
                reduce(lambda acc, txn: acc + i.issubset(set(txn)), txns, 0)
                for i in self.itemsets
            ]
            return DataMiner.AprioriTable(self.itemsets, supports)

        def frequents(self, sup_min: int = 0) -> DataMiner.AprioriTable:
            pairs = [x for x in zip(
                self.itemsets, self.supports) if x[1] >= sup_min]
            return DataMiner.AprioriTable(
                *(zip(*pairs) if len(pairs) > 0 else ([], []))
            )

        def infrequents(self, sup_min: int = 0) -> DataMiner.AprioriTable:
            pairs = [x for x in zip(
                self.itemsets, self.supports) if x[1] < sup_min]
            return DataMiner.AprioriTable(
                *(zip(*pairs) if len(pairs) > 0 else ([], []))
            )

        def join(self) -> DataMiner.AprioriTable:
            size = len(self.itemsets)
            if size > 0:
                joined = []
                level = len(self.itemsets[0])
                joined = [x.union(y)
                          for x in self.itemsets for y in self.itemsets]
                joined = [x for x in joined if len(x) == level + 1]
                joined = reduce(
                    lambda l, v: l if v in l else l + [v], joined, [])
                return DataMiner.AprioriTable(joined)
            else:
                return DataMiner.AprioriTable([])

        def prune(self, filtered: DataMiner.AprioriTable) -> DataMiner.AprioriTable:
            pruned = [
                i
                for i in self.itemsets
                if not any(map(lambda v: v.issubset(i), filtered.itemsets))
            ]
            return DataMiner.AprioriTable(pruned)

        def to_dict(self) -> defaultdict[tuple[int], int]:
            dictionary = defaultdict(int)
            for k, v in zip(self.itemsets, self.supports):
                key = tuple(sorted(k))
                dictionary[key] = v
            return dictionary

        def __len__(self):
            return len(self.itemsets)

        def __add__(self, at: DataMiner.AprioriTable) -> DataMiner.AprioriTable:
            if len(self.itemsets) != len(self.supports):
                raise AttributeError('Left table isn\'t scanned yet')
            if len(at.itemsets) != len(at.supports):
                raise AttributeError('Right table isn\'t scanned yet')
            return DataMiner.AprioriTable(self.itemsets + at.itemsets, self.supports + at.supports)

    def __init__(self, dataset: str, sup_min_prob: float, k: int = 5):
        self.dataloader = DataLoader(dataset)
        self.sup_min_prob = sup_min_prob
        self.ntxns = len(self.dataloader)
        self.sup_min = int(self.sup_min_prob * self.ntxns)
        self.k = k
        assert self.ntxns % self.k == 0

    def _apriori(self, partition: list[list[int]]) -> defaultdict[tuple[int], int]:
        sup_min = self.sup_min // self.k
        frequents = DataMiner.AprioriTable([])
        itemsets = set()
        for txn in partition:
            itemsets = itemsets.union(set(txn))
        itemsets = [{v} for v in itemsets]
        C = DataMiner.AprioriTable(itemsets).scan(partition)
        L = C.frequents(sup_min)
        frequents += L
        while len(L) > 0:
            C = L.join().prune(C.infrequents(sup_min)).scan(partition)
            L = C.frequents(sup_min)
            frequents += L
        return frequents.to_dict()

    def prepare(self) -> dict[tuple[int], int]:
        print('>>> prepare()')
        candidates = defaultdict(int)
        partition = []
        for _ in tqdm(range(0, self.k), desc="First Scan"):
            partition = self.dataloader.batch(self.ntxns // self.k)
            local_frequents = self._apriori(partition)
            for c in local_frequents:
                candidates[c] = 0
        self.dataloader.seek(0)

        for txn in tqdm(self.dataloader, desc="Second Scan"):
            for c in candidates:
                if set(c).issubset(set(txn)):
                    candidates[c] += 1

        frequents = dict()
        for c in candidates:
            if candidates[c] >= self.sup_min:
                frequents[c] = candidates[c]
        return frequents

    def mining(
        self, itemsets: dict[tuple[int], int]
    ) -> list[tuple[tuple[int], tuple[int], float, float]]:
        print('>>> mining()')
        patterns = []
        assoc = [(l, r)
                 for l in itemsets for r in itemsets if set(r).isdisjoint(set(l))]
        for l, r in tqdm(assoc, desc="Mining association rules"):
            u = tuple(sorted(set(l).union(set(r))))
            try:
                sup_l = itemsets[l]
                sup_u = itemsets[u]
                sup_prob = sup_u / self.ntxns
                conf_prob = sup_u / sup_l
                if sup_prob >= self.sup_min_prob:
                    patterns.append((l, r, sup_prob * 100, conf_prob * 100))
            except:
                pass
        return patterns

    def log(self, fout: str, patterns: list[tuple[tuple[int], tuple[int], float, float]]) -> None:
        with open(fout, "w") as f:
            for l, r, support, confidence in patterns:
                l_str = f"{{{','.join(map(str, l))}}}"
                r_str = f"{{{','.join(map(str, r))}}}"
                f.write(f"{l_str}\t{r_str}\t{support:.2f}\t{confidence:.2f}\n")


def main():
    # Parse sys argv
    if len(sys.argv) != 4:
        required = (
            "{minimum support probablity: float} {input file: str} {output file: str}"
        )
        print(f"Usage: {sys.argv[0]} {required}")
        sys.exit()
    sup_min_prob = float(sys.argv[1]) / 100
    fin = str(sys.argv[2])
    fout = str(sys.argv[3])

    # DataMining algorithm
    miner = DataMiner(fin, sup_min_prob)
    itemsets = miner.prepare()
    patterns = miner.mining(itemsets)
    miner.log(fout, patterns)


if __name__ == "__main__":
    main()
