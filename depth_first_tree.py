

class Node:
    def __init__(self,val):
        self.right = None
        self.left = None
        self.value = val


def dfs_tree_search(node,search_val):
    if node is None:
        return False
    if node.value == search_val:
        return True
    left_side = dfs_tree_search(node.left,search_val)
    if left_side:
        return True
    right_side = dfs_tree_search(node.right, search_val)
    if right_side:
        return True
    return False

if __name__ == '__main__':
    tree = Node(6)
    tree.left = Node(3)
    tree.right = Node(4)
    tree.left.left = Node(2)
    tree.left.right = Node(1)
    tree.right.left = Node(7)
    print(dfs_tree_search(tree, 10))
