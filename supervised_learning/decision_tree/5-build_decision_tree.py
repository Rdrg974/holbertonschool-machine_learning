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
        self.lower = {}
        self.upper = {}

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

    def left_child_add_prefix(self, text):
        """
        Adds a prefix for the left child in the string representation.

        Parameters:
        - text (str): The text representation of the child node.

        Returns:
        - str: The text with the appropriate prefix for the left child.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Adds a prefix for the right child in the string representation.

        Parameters:
        - text (str): The text representation of the child node.

        Returns:
        - str: The text with the appropriate prefix for the right child.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        if new_text.endswith("\n"):
            new_text = new_text[:-1]
        return new_text

    def __str__(self):
        """
        Returns a string representation of the tree structure
        starting from the current node.

        Returns:
        - str: A string representation of the node and its subtrees.
        """
        result = f"[feature={self.feature}, threshold={self.threshold}]\n"
        if self.is_root:
            result = "root " + result
        else:
            result = "-> node " + result

        if self.left_child:
            result += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            result += self.right_child_add_prefix(str(self.right_child))

        if self.depth == 0:
            result += "\n"
        return result

    def get_leaves_below(self):
        """
        Returns the list of all leaves of the tree

        Returns:
        - str: A string representation of list of all leaves of the tree
        """
        leaves = []
        if self.is_leaf:
            leaves.append(self)
        else:
            if self.left_child:
                leaves.extend(self.left_child.get_leaves_below())
            if self.right_child:
                leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Updates the lower and upper bounds for the node and its children.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()

                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                elif child == self.right_child:
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):
        """
        Compute the indicator function for
        the node based on its lower and upper bounds.
        """
        def is_large_enough(x):
            """
            Check if all features are greater than or equal to lower bounds.
            """
            check = np.array([
                np.greater(x[:, key], self.lower[key])
                for key in self.lower.keys()
            ])
            return np.all(check, axis=0)

        def is_small_enough(x):
            """
            Check if all features are less than or equal to upper bounds.
            """
            check = np.array([
                np.less_equal(x[:, key], self.upper[key])
                for key in self.upper.keys()
            ])
            return np.all(check, axis=0)

        self.indicator = lambda x: np.all(np.array([
            is_large_enough(x), is_small_enough(x)
        ]), axis=0)


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
        """
        Counts only the leaf node itself, since it doesn't have children.

        Parameters:
        - only_leaves (bool): This parameter is ignored
        for leaf nodes since it will always count the leaf.

        Returns:
        - int: Always returns 1 as it counts the leaf node itself.
        """
        return 1

    def __str__(self):
        """
        Returns a string representation of the leaf node.

        Returns:
        - str: A string representation in the format
        '-> leaf [value={value}]' where value is the value of the leaf.
        """
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """
        Returns the list of all leaves of the tree

        Returns:
        - str: A string representation of list of all leaves of the tree
        """
        return [self]

    def update_bounds_below(self):
        """
        Pass
        """
        pass


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
        """
        Counts the nodes in the tree, starting from the root node.

        Parameters:
        - only_leaves (bool): If True, counts only leaf nodes.
        If False, counts all nodes.

        Returns:
        - int: The total number of nodes in the tree .
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns a string representation of the entire
        decision tree starting from the root.

        Returns:
        - str: A string representation of the tree,
        beginning with the root node and recursively showing its children.
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Returns the list of all leaves of the tree

        Returns:
        - str: A string representation of list of all leaves of the tree
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Updates the lower and upper bounds for the node and its children.
        """
        self.root.update_bounds_below()
