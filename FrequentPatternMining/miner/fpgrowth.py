import pandas as pd
import numpy as np
import math
from collections import defaultdict
from itertools import chain, combinations


class Node(object):
    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = defaultdict(Node)
        if parent is not None:
            parent.children[item] = self

    def branch_from_root(self):
        path = []
        if self.item is None:
            return path
        node = self.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent
        path.reverse()
        return path

    def __repr__(self):
        return f'Node[{self.item}](count: {self.count})'


class FPTree(object):
    def __init__(self, rank=None):
        self.root = Node(None)
        self.nodes = defaultdict(list)
        self.cond_items = []
        self.rank = rank

    def conditional_tree(self, cond_item, min_sup: int):
        branches = []
        count = defaultdict(int)
        for node in self.nodes[cond_item]:
            branch = node.branch_from_root()
            branches.append(branch)
            for item in branch:
                count[item] += node.count

        items = [item for item in count if count[item] >= min_sup]
        items.sort(key=count.get)
        rank = {item: i for i, item in enumerate(items)}

        cond_tree = FPTree(rank)
        for idx, branch in enumerate(branches):
            branch = sorted([i for i in branch if i in rank],
                            key=rank.get, reverse=True)
            cond_tree.insert(branch, self.nodes[cond_item][idx].count)
        cond_tree.cond_items = self.cond_items + [cond_item]

        return cond_tree

    def insert(self, frequents: list, count: int = 1):
        self.root.count += count
        node = self.root
        for item in frequents:
            if item in node.children:
                child = node.children[item]
                child.count += count
                node = child
            else:
                child = Node(item, count, node)
                self.nodes[item].append(child)
                node = child

    def is_path(self):
        node = self.root
        while len(node.children) > 0:
            if len(node.children) == 1:
                item = list(node.children.keys())[0]
                node = node.children[item]
            else:
                return False
        return True


def setup_fptree(tdb: np.ndarray[np.ndarray[bool]], min_sup: int):
    supports = np.array(np.sum(tdb, axis=0)).reshape(-1)
    items = np.nonzero(supports >= min_sup)[0]
    indices = supports[items].argsort()
    rank = {item: i for i, item in enumerate(items[indices])}
    tree = FPTree(rank)
    for i in range(len(tdb)):
        itemset = np.where(tdb[i])[0]
        frequents = [item for item in itemset if item in rank]
        frequents.sort(key=rank.get, reverse=True)
        tree.insert(frequents)
    return tree


def procedure(tree: FPTree, min_sup: int):
    items = tree.nodes.keys()
    if tree.is_path():
        frequents = chain.from_iterable(combinations(items, r)
                                        for r in range(1, len(items) + 1))
        for itemset in frequents:
            support = min([tree.nodes[item][0].count for item in itemset])
            yield tree.cond_items + list(itemset), support
    else:
        for item in items:
            support = sum([node.count for node in tree.nodes[item]])
            yield tree.cond_items + [item], support
            cond_tree = tree.conditional_tree(item, min_sup)
            for itemset, support in procedure(cond_tree, min_sup):
                yield itemset, support


def mine(df: pd.DataFrame, min_support: float) -> pd.DataFrame:
    items = df.columns
    tdb = df.values.astype(np.int64)
    min_sup = math.ceil(min_support * len(tdb))
    tree = setup_fptree(tdb, min_sup)
    itemsets = []
    supports = []
    for itemset, support in procedure(tree, min_sup):
        itemsets.append(itemset)
        supports.append(support / len(tdb))
    res_df = pd.DataFrame({"itemset": itemsets, "support": supports})
    res_df["itemset"] = res_df["itemset"].apply(
        lambda x: sorted([items[i] for i in x])
    )
    return res_df
