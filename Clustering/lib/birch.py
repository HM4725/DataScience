import numpy as np

BRANCHING_FACTOR = 100
THRESHOLD = 0.3


def distance(d1, d2):
    return np.abs(d1 - d2).sum()


def cf2centroid(cf):
    n, LS, _ = cf
    return LS / n


def cf2radius(cf):
    n, LS, SS = cf
    return np.sqrt((n * SS - 2 * LS ** 2 + n * LS) / pow(n, 2))


class InternalNode:
    def __init__(self):
        self.parent = None
        self.key = -1
        self.cfs = [None for _ in range(BRANCHING_FACTOR)]
        self.children = [None for _ in range(BRANCHING_FACTOR)]
        self.sz = 0

    def insert_leaf(self, leaf):
        self.cfs[self.sz] = [0, np.array([0, 0]), np.array([0, 0])]
        self.children[self.sz] = leaf
        leaf.parent = self
        leaf.key = self.sz
        self.sz += 1

    def insert_datapoint(self, d, entry):
        cf = self.cfs[entry]
        cf[0] += 1
        cf[1] += d
        cf[2] += d**2


class LeafNode:
    def __init__(self):
        self.parent = None
        self.key = -1
        self.cfs = [None for _ in range(BRANCHING_FACTOR)]
        self.sz = 0
        self.prev = None
        self.next = None

    def alloc_new_entry(self):
        entry = self.sz
        self.cfs[entry] = [0, np.array([0, 0]), np.array([0, 0])]
        self.sz += 1
        return entry

    def insert_datapoint(self, d, entry):
        cf = self.cfs[entry]
        cf[0] += 1
        cf[1] += d
        cf[2] += d**2

    def split(self):
        left = LeafNode()
        left.cfs = self.cfs[:self.sz//2]
        left.sz = self.sz//2
        left.parent = self.parent
        right = LeafNode()
        right.cfs = self.cfs[self.sz//2:]
        right.sz = self.sz - left.sz
        right.parent = self.parent

        return left, right


class CFTree:
    def __init__(self, branching_factor=100, threshold=0.5):
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.root = InternalNode()

    def __insert_first(self, d):
        node = LeafNode()
        self.root.insert_leaf(node)
        entry = node.alloc_new_entry()
        node.insert_datapoint(d, entry)

    def __search_closest_leaf(self, d):
        node = self.node
        while isinstance(node, InternalNode):
            node = node.children[0]
        leaf = node
        closest_leaf = None
        min_dist = float('inf')
        entry = -1
        while leaf != None:
            # check closest_node
            for i in range(node.sz):
                cf = node.cfs[i]
                ctrd = cf2centroid(cf)
                dist = distance(d, ctrd)
                if dist < min_dist:
                    min_dist = dist
                    closest_leaf = leaf
                    entry = i
            leaf = leaf.next
        return closest_leaf, entry

    def __insert(self, d):
        leaf, entry = self.__search_closest_leaf(d)
        cf = leaf.cfs[entry]
        cf_next = [cf[0] + 1, cf[1] + d, cf[2] + d**2]
        radius = cf2radius(cf_next)
        if radius < THRESHOLD:
            leaf.cfs[entry] = cf_next
        elif leaf.sz < BRANCHING_FACTOR - 1:
            entry = leaf.alloc_new_entry()
            leaf.insert_datapoint(d, entry)
        else: #TODO: split node
            left, right = leaf.split()
            pass
        
        node = leaf
        while node.parent != None:
            entry = node.key
            node.parent.insert_datapoint(d, entry)
            node = node.parent



    def insert(self, d):
        if self.root.sz == 0:
            self.__insert_first(d)
        else:
            self.__insert(d)
