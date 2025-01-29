#!/usr/bin/env python3

"""
Module for decision tree management with implementations of Node, Leaf,
and Decision_Tree classes. This module allows the creation of a decision tree
with methods to compute the maximum depth of subtrees.
"""
import numpy as np


class Node:
    """
    Class representing a node in a decision tree.

    Attributes:
    - feature: The feature used for splitting the node.
    - threshold: The threshold used for splitting the node.
    - left_child: Left child node.
    - right_child: Right child node.
    - is_leaf: Boolean indicating if the node is a leaf.
    - is_root: Boolean indicating if the node is the root.
    - sub_population: Data associated with the node.
    - depth: The depth of the node in the tree.
    """
    def __init__(
        self, feature=None, threshold=None, left_child=None, right_child=None,
        is_root=False, depth=0
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Computes the maximum depth of the subtrees
        starting from the current node.

        Returns:
        - The maximum depth among the left and right subtrees.
        """
        if self.is_leaf:
            return self.depth
        left_depth = (
            self.left_child.max_depth_below() if self.left_child else -1
        )
        right_depth = (
            self.right_child.max_depth_below() if self.right_child else -1
        )
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the nodes below the current node.

        Parameters:
        - only_leaves (bool): If True, counts only leaf nodes.
        If False, counts all nodes.

        Returns:
        - The number of nodes below this node (leaf or non-leaf).
        """
        count = 1 if not only_leaves or self.is_leaf else 0
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)

        return count


class Leaf(Node):
    """
    Class representing a leaf in a decision tree.

    Attributes:
    - value: The value of the leaf.
    - depth: The depth of the leaf in the tree.
    """
    def __init__(self, value, depth=None):
        """
        Initializes a leaf with a value and depth.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf.

        Returns:
        - The depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1


class Decision_Tree():
    """
    Class representing a decision tree.

    Attributes:
    - max_depth: Maximum depth of the tree.
    - min_pop: Minimum size of sub-populations in nodes.
    - seed: Seed for random number generation.
    - split_criterion: Criterion for splitting nodes.
    - root: Root node of the tree.
    """
    def __init__(
        self, max_depth=10, min_pop=1, seed=0, split_criterion="random",
        root=None
    ):
        """
        Initializes a decision tree with the given parameters.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Returns the maximum depth of the tree starting from the root.

        Returns:
        - The maximum depth of the root subtree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)
