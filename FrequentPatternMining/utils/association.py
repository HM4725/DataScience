import pandas as pd
from itertools import product


def association_rule(patterns: pd.DataFrame, min_support: float, min_conf: float):
    itemsets = patterns['itemset']
    supports = patterns['support']
    mapper = {tuple(i): s for i, s in zip(itemsets, supports)}

    assoc = filter(lambda x: set(x[1]).isdisjoint(
        set(x[0])), product(itemsets, itemsets))
    for l, r in assoc:
        u = sorted(set(l).union(set(r)))
        sup_l = mapper[tuple(l)]
        sup_u = mapper.get(tuple(u), 0)
        if sup_u > 0:
            support = sup_u
            confidence = sup_u / sup_l
            if support >= min_support and confidence >= min_conf:
                yield l, r, support, confidence
