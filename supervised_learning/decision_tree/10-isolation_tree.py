#!/usr/bin/env python3
"""
This module contains the class Isolation Random Tree
"""

import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """Isolation Random Tree for outlier detection."""
    def __init__(self, max_depth=10, seed=0, root=None):
        """Initialize the Isolation Random Tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Return a string representation of the tree."""
        return self.root.__str__()

    def depth(self):
        """Return the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes or leaves in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Update the bounds of all nodes in the tree."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Return all leaf nodes of the tree."""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Update the prediction function for the tree."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        if not leaves:
            self.predict = lambda A: np.array([])
        else:
            self.predict = lambda A: np.array([
                leaf.depth for leaf in leaves
            ])[
                np.argmax([leaf.indicator(A) for leaf in leaves], axis=0)
            ]

    def np_extrema(self, arr):
        """Find minimum and maximum values in an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Generate a random split criterion for a node."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Create a leaf child node."""
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create an internal child node."""
        node_child = Node()
        node_child.depth = node.depth + 1
        node_child.sub_population = sub_population
        return node_child

    def fit_node(self, node):
        """Recursively fit a node and its children."""
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] <= node.threshold
        )

        # Is left node a leaf?
        is_left_leaf = (
            np.sum(left_population) <= 1 or node.depth + 1 == self.max_depth
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf?
        is_right_leaf = (
            np.sum(right_population) <= 1 or node.depth + 1 == self.max_depth
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Fit the tree to the given explanatory data."""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones_like(
            explanatory.shape[0], dtype='bool'
        )

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
