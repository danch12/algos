from collections import deque


class Node:
    __slots__ = 'val', 'left', 'right'

    def __init__(self, val, left=None, right=None):
        self.val = val
        self.right = right
        self.left = left


def breadth_first_search(root):
    queue = deque()

    queue.append(root)
    while len(queue) > 0:
        node = queue.popleft()
        print(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)


def average_of_levels(root):
    if root is None:
        return []
    averages = []
    queue = deque()
    temp = deque()
    queue.append(root)
    level_sum = 0
    count = 0
    while len(queue) > 0 or len(temp) > 0:
        node = queue.popleft()
        level_sum += node.val
        count += 1
        if node.left:
            temp.append(node.left)
        if node.right:
            temp.append(node.right)
        if len(queue) == 0:
            averages.append(level_sum / count)
            level_sum = 0
            count = 0
            queue = temp
            temp = deque()

    return averages


def minDepth(self, root):
    if root is None:
        return 0
    if root.left:
        left = self.minDepth(root.left)
    if root.right:
        right = self.minDepth(root.right)

    if not root.left and not root.right:
        return 1
    if not root.left:
        return 1 + right
    if not root.right:
        return 1 + left

    return 1 + min(left, right)


def is_same_tree(p, q):
    if p is None and q is None:
        return True
    if p is None and q is not None:
        return False
    if q is None and p is not None:
        return False

    def dfs(root_p, root_q):
        if not root_p and not root_q:
            return True

        if not root_p or not root_q:
            return False

        if root_p.val != root_q.val:
            return False

        return dfs(root_q.left, root_p.left) and dfs(root_q.right, root_p.right)

    return dfs(p, q)


def has_path_sum(root, target_sum):
    def dfs_sum(root, target_sum, current):

        current += root.val
        if root.left is None and root.right is None:
            return current == target_sum
        if root.right:
            right = dfs_sum(root.right, target_sum, current)
        else:
            right = False
        if root.left:
            left = dfs_sum(root.left, target_sum, current)
        else:
            left = False
        return right or left

    if root is None:
        return False
    return dfs_sum(root, target_sum, 0)


def invertTree(root):
    def reverse(root):
        if root is not None:
            left = reverse(root.left)
            right = reverse(root.right)
            root.left = right
            root.right = left
            return root
        return None

    reverse(root)
    return root


def mergeTrees(root1, root2):
    if root1 == None:
        return root2
    if root2 == None:
        return root1
    root1.val += root2.val
    root1.left = mergeTrees(root1.left, root2.left)
    root1.right = mergeTrees(root1.right, root2.right)
    return root1


def diameter_of_binary_tree(root):
    def max_height(root, best):
        if root is None:
            return 0
        left = max_height(root.left, best)
        right = max_height(root.right, best)
        best[0] = max(best[0], left + right)
        return 1 + max(left, right)

    diameter = [0]
    max_height(root, diameter)
    return diameter[0]


def lowestCommonAncestor(root, p, q):
    if max(p.val, q.val) < root.val:
        return lowestCommonAncestor(root.left, p, q)
    elif min(p.val, q.val) > root.val:
        return lowestCommonAncestor(root.right, p, q)
    else:
        return root


def maxDepth(root):
    if root is None:
        return 0

    return 1 + max(maxDepth(root.left), maxDepth(root.right))


def cloneGraph(node):
    if node is None:
        return node
    visited = {node: Node(node.val)}

    def dfs(node, visited):

        for neighbor in node.neighbors:
            if neighbor not in visited:
                visited[neighbor] = Node(neighbor.val)
                dfs(neighbor, visited)
            visited[node].neighbors.append(visited[neighbor])

    dfs(node, visited)
    return visited[node]


def dfs_array(matrix, i, j, visited, m, n):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    # when dfs called, meaning its caller already verified this point
    visited[i][j] = True
    for direction in directions:
        x, y = i + direction[0], j + direction[1]
        if x < 0 or x >= m or y < 0 or y >= n or visited[x][y] or matrix[x][y] < matrix[i][j]:
            continue
        dfs_array(matrix, x, y, visited, m, n)


def numIslands(grid):
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def dfs(matrix, i, j, visited, m, n):
        visited[i][j] = True
        for direction in directions:
            y, x = i + direction[0], j + direction[1]
            if y < 0 or x < 0 or y >= m or x >= n or grid[y][x] == '0' or visited[y][x]:
                continue
            dfs(matrix, y, x, visited, m, n)

    m, n = len(grid), len(grid[0])
    visited = [[False for _ in range(n)] for _ in range(m)]
    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1' and not visited[i][j]:
                dfs(grid, i, j, visited, m, n)
                count += 1
    return count


def zigzagLevelOrder(self, root):
    if root is None:
        return []
    dq = deque([root])
    res = []
    lvl = 0
    while dq:
        size = len(dq)
        curr = []
        for i in range(size):
            if lvl % 2 == 0:
                p = dq.popleft()
                curr.append(p.val)
                if p.left:
                    dq.append(p.left)
                if p.right:
                    dq.append(p.right)
            else:
                p = dq.pop()
                curr.append(p.val)
                if p.right:
                    dq.appendleft(p.right)
                if p.left:
                    dq.appendleft(p.left)
        res.append(curr)
        lvl += 1
    return res


"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""


def connect(root):
    if not root:
        return

    queue = deque()
    temp = deque()

    queue.append(root)

    while len(queue) > 0 or len(temp) > 0:
        node = queue.popleft()
        if node.left:
            temp.append(node.left)
        if node.right:
            temp.append(node.right)

        if len(queue) == 0:
            queue = temp
            temp = deque()
        else:
            node.next = queue[0]
    return root


def connect_no_space(root):
    node = root
    while node:
        curr = dummy = Node(0)
        while node:
            if node.left:
                curr.next = node.left
                curr = curr.next

            if node.right:
                curr.next = node.right
                curr = curr.next
            node = node.next
        node = dummy.next

    return root


# get right side of binary tree
def rightSideView(root):
    def right_helper(root, depth):
        if root:
            if depth == len(vals):
                vals.append(root.val)
            right_helper(root.right, depth + 1)
            right_helper(root.left, depth + 1)

    vals = []

    right_helper(root, 0)
    return vals


# find nodes k distance away from target
def distanceK(root, target, k):
    # connect them all up like a graph
    def dfs(node, parent=None):
        if node:
            node.parent = parent
            if node.left:
                dfs(node.left, node)
            if node.right:
                dfs(node.right, node)

    answer = []

    def search(start, k, seen={}):

        if start in seen:
            return
        seen[start] = start
        if k < 0:
            return
        if k == 0:
            answer.append(start.val)
        if start.parent:
            search(start.parent, k - 1, seen)
        if start.left:
            search(start.left, k - 1, seen)
        if start.right:
            search(start.right, k - 1, seen)

    dfs(root)
    search(target, k)
    return answer


def pathSum(root, target_sum):
    def helper(root, target_sum, path):
        if not root:
            return

        if target_sum - root.val == 0 and not root.left and not root.right:
            path += [root.val]
            ans.append(path)
            return
        if root.left:
            helper(root.left, target_sum - root.val, path + [root.val])
        if root.right:
            helper(root.right, target_sum - root.val, path + [root.val])

    ans = []
    helper(root, target_sum, [])
    return ans


class Solution:
    ans = 0

    def pathSum(self, root, targetSum):
        def dfs(root, cache, curr_path_sum, target):
            if not root:
                return

            curr_path_sum += root.val
            old_path_sum = curr_path_sum - target
            self.ans += cache.get(old_path_sum, 0)
            cache[curr_path_sum] = cache.get(curr_path_sum, 0) + 1

            dfs(root.left, cache, curr_path_sum, target)

            dfs(root.right, cache, curr_path_sum, target)

            cache[curr_path_sum] -= 1

        dfs(root, {0: 1}, 0, targetSum)
        return self.ans


class Solution2:
    ans = None

    def lowestCommonAncestor(self, root, p, q):
        def helper(root, p, q):
            if not root:
                return False

            mid = False
            if root == p or root == q:
                mid = True

            left = helper(root.left, p, q)
            right = helper(root.right, p, q)

            if (mid and left) or (mid and right) or (left and right):
                self.ans = root

            return mid or left or right

        helper(root, p, q)
        return self.ans


class Solution3:
    def constructMaximumBinaryTree(self, nums):
        def find_largest_index(nums):
            max_num = nums[0]
            max_ind = 0
            for i in range(0, len(nums)):
                if nums[i] > max_num:
                    max_num = nums[i]
                    max_ind = i
            return max_ind

        def create_tree(nums):
            if len(nums) == 0:
                return None

            ind = find_largest_index(nums)
            left = nums[:ind]
            right = nums[ind + 1:]

            return Node(nums[ind], create_tree(left), create_tree(right))

        return create_tree(nums)


def widthOfBinaryTree(root):
    if not root: return []
    width, level = 0, [(root, 1)]
    # Keep going untill current level has some nodes.
    while len(level):
        # Put all children of current level in next_level.
        width = max(width, level[-1][1] - level[0][1] + 1)
        next_level = []
        for item, num in level:
            if item.left:  # Make sure to not put the Null nodes
                next_level.append((item.left, num * 2))
            if item.right:
                next_level.append((item.right, num * 2 + 1))
        level = next_level
    return width

#build tree from preorder and inorder arrays
class Solution:
    preorder_index = 0

    def buildTree(self, preorder, inorder):

        def find_indexes(nums):
            dic = {}
            for ind, num in enumerate(nums):
                dic[num] = ind
            return dic

        def build_tree(left, right):
            if left > right:
                return None

            root_value = preorder[self.preorder_index]
            root = Node(root_value)

            self.preorder_index += 1

            root.left = build_tree(left, inorder_pos[root_value] - 1)
            root.right = build_tree(inorder_pos[root_value] + 1, right)
            return root

        inorder_pos = find_indexes(inorder)
        return build_tree(0, len(preorder) - 1)


def isValidBST(root):
    def check(root, lessthan=float('inf'), largerthan=float('-inf')):

        if (root.val <= largerthan) or (root.val >= lessthan):
            return False

        left = True
        if root.left:
            left = check(root.left, root.val, largerthan)

        right = True
        if root.right:
            right = check(root.right, lessthan, root.val)

        return left and right

    return check(root)


if __name__ == '__main__':
    node = Node(1)
    node.left = Node(2)
    node.right = Node(3)
    node.right.left = Node(4)
    avs = average_of_levels(node)
    breadth_first_search(node)
    print(avs)
