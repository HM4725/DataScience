from __future__ import annotations
from .dataframe import DataFrame
from collections import defaultdict
from . import entropy


class DTNode(defaultdict):
    feature: str
    decision: str

    def __init__(self, feature: str, options: list[str] = []):
        super().__init__(DTNode)
        self.feature = feature
        self.decision = None
        for opt in options:
            self[opt] = None

    def leafify(self, decision: str):
        self.decision = decision

    def predict(self, choices: dict[str, str]) -> str:
        if self.decision != None:
            return self.decision
        else:
            opt = choices.pop(self.feature)
            return self[opt].predict(choices)

    def traverse(self):
        """Breadth First Search
        """
        level = [self, None]
        while len(level) > 0:
            next_level = []
            for i, n in enumerate(level):
                if n == None:
                    print("|| ", end="")
                elif isinstance(n, DTNode):
                    print(n, end=", " if level[i+1] != None else " ")
                    if n.decision == None:
                        next_level.extend(n.values())
                        next_level.append(None)
            print()
            level = next_level

    def __repr__(self):
        if self.decision == None:
            return f"Node({self.feature}) {{{', '.join(self.keys())}}}"
        else:
            return f"<{self.decision}>"


def gain(df: DataFrame, feature: str):
    feature_label = df.columns[-1]

    # Total dataset (D)
    len_D = len(df)
    classes = list(map(lambda x: len(x), df.groupby(
        feature_label, keep=True).values()))
    info_D = entropy.H(classes)

    # Conditional dataset (D|F)
    res = info_D
    groups = df.groupby(feature)
    weights = [len(index) / len_D for index in groups.values()]

    for i, attr in enumerate(groups):
        df_F = df[groups[attr]]
        classes = list(map(lambda x: len(x), df_F.groupby(
            feature_label, keep=True).values()))
        res -= (weights[i] * entropy.H(classes))
    if res == 0:
        return 0
    else:
        split = entropy.H(weights)
        return res / split


def construct_tree(df: DataFrame, parent: DTNode = None, parent_attr: str = None):
    features = df.columns[:-1]
    feature_label = df.columns[-1]

    # Select feature to split data
    gains = [gain(df, feat) for feat in features]
    argmax = max(range(len(features)), key=lambda x: gains[x])
    feature = features.pop(argmax)
    attributes = df.attributes[argmax]
    node = DTNode(feature, attributes)
    if parent != None:
        parent[parent_attr] = node

    # Selected feature
    selected = df.groupby(feature, keep=True)
    for attr in selected:
        df_child = df[selected[attr]][features + [feature_label]]
        labels = df_child.groupby(feature_label)
        # Classify
        label = None
        if len(labels) == 1:
            # Perfectly classify
            label = list(labels.keys())[0]
        elif len(labels) == 0:
            # Fail to classify: Use upper D instead D|F
            labels = df.groupby(feature_label)
            label = max(labels.keys(), key=lambda x: len(labels[x]))
        elif len(df_child.columns) == 1:
            # Majority voting
            label = max(labels.keys(), key=lambda x: len(labels[x]))

        # Stop or go
        if label != None:
            # stop
            leaf = DTNode(feature_label)
            leaf.leafify(label)
            node[attr] = leaf
        else:
            # Go: Divide and conquer
            construct_tree(df_child, node, attr)
    return node


def dt_fit(df: DataFrame) -> DTNode:
    return construct_tree(df)


def dt_predict(tree: DTNode, df: DataFrame, feature_label: str) -> DataFrame:
    predicted = [tree.predict({f: a for f, a in zip(
        df.columns, df.row(m))}) for m in range(len(df))]
    return df + DataFrame({feature_label: predicted})
