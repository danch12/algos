# quite similar to dfs
def exist(board, word):
    def helper(row, col, i, board):

        if row < 0 or col < 0:
            return False

        if row >= len(board) or col >= len(board[0]):
            return False

        if i == len(word) - 1 and board[row][col] == word[i]:
            return True

        if board[row][col] != word[i]:
            return False

        tmp = board[row][col]
        board[row][col] = '#'

        up = helper(row - 1, col, i + 1, board)
        if up == True:
            return True

        down = helper(row + 1, col, i + 1, board)
        if down == True:
            return True

        left = helper(row, col - 1, i + 1, board)
        if left == True:
            return True

        right = helper(row, col + 1, i + 1, board)
        if right == True:
            return True

        board[row][col] = tmp
        return False

    for i in range(len(board)):
        for j in range(len(board[0])):
            if helper(i, j, 0, board):
                return True
    return False


def letterCasePermutation(s):
    ans = []

    def helper(i, current_str):
        if i == len(s):
            ans.append(current_str)
            return

        if s[i].isalpha():
            helper(i + 1, current_str + s[i].upper())
            helper(i + 1, current_str + s[i].lower())
        else:
            helper(i + 1, current_str + s[i])

    helper(0, '')
    return ans


# Given an integer array nums of unique elements, return all possible subsets (the power set).
# The solution set must not contain duplicate subsets. Return the solution in any order.
def subsets(nums):
    subsets = []

    def helper(current_subset, first):
        subsets.append(current_subset[:])

        for i in range(first, len(nums)):
            current_subset.append(nums[i])
            helper(current_subset, i + 1)
            current_subset.pop()

    helper([], 0)
    return subsets


def subsetsWithDup(nums):
    def helper(current_subset, current_num):
        subset.append(current_subset[:])
        for i in range(current_num, len(nums)):
            if i > current_num and nums[i] == nums[i - 1]:
                continue

            current_subset.append(nums[i])
            helper(current_subset, i + 1)
            current_subset.pop()

    subset = []
    nums.sort()
    helper([], 0)
    return subset


def permute(nums):
    def backtrack(curr_perm, nums):
        if not nums:
            ans.append(curr_perm[:])
            return

        for i in range(0, len(nums)):
            curr_perm.append(nums[i])
            backtrack(curr_perm, nums[:i] + nums[i + 1:])
            curr_perm.pop()

    ans = []
    backtrack([], nums)
    return ans


def permuteUnique(nums):
    counter = {}
    for num in nums:
        if num in counter:
            counter[num] += 1
        else:
            counter[num] = 1

    def backtrack(curr_perm, counter):
        if len(curr_perm) == len(nums):
            ans.append(curr_perm[:])
            return

        for key in counter.keys():
            if counter[key] > 0:
                curr_perm.append(key)
                counter[key] -= 1
                backtrack(curr_perm, counter)
                counter[key] += 1
                curr_perm.pop()

    ans = []
    backtrack([], counter)
    return ans


# Given two integers n and k, return all possible combinations of k numbers out of the range [1, n].
def combine(n: int, k: int):
    def backtrack(curr, ind):
        if len(curr) == k:
            ans.append(curr[:])
            return

        for i in range(ind, n + 1):
            curr.append(i)
            backtrack(curr, i + 1)
            curr.pop()

    ans = []

    backtrack([], 1)
    return ans


def combinationSum(candidates, target):
    def backtrack(curr_group, curr_sum, candidates):
        if curr_sum == target:
            ans.append(curr_group[:])
            return
        if curr_sum > target:
            return

        for i in range(len(candidates)):
            backtrack(curr_group + [candidates[i]], curr_sum + candidates[i], candidates[i:])

    ans = []
    backtrack([], 0, candidates)
    return ans


def combinationSum2(candidates, target: int):
    def backtrack(current_group, current_total, start):
        if current_total > target:
            return
        if current_total == target:
            ans.append(current_group[:])
            return

        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            backtrack(current_group + [nums[i]], current_total + nums[i], i + 1)

    nums = sorted(candidates)
    ans = []
    backtrack([], 0, 0)
    return ans


def combinationSum3(k: int, n: int):
    def backtrack(curr_group, curr_sum, start):
        if len(curr_group) == k and curr_sum == n:
            ans.append(curr_group[:])
            return

        if len(curr_group) > k:
            return

        if curr_sum > n:
            return

        for i in range(start, 10):
            backtrack(curr_group + [i], curr_sum + i, i + 1)

    ans = []

    backtrack([], 0, 1)
    return ans


def generateParenthesis(self, n: int):
    def backtrack(brackets, opened, closed):
        if closed > opened:
            return

        if opened == n and closed == n:
            ans.append(brackets)
            return

        if opened > n:
            return

        if closed > n:
            return

        backtrack(brackets + '(', opened + 1, closed)
        backtrack(brackets + ')', opened, closed + 1)

    ans = []
    backtrack('', 0, 0)
    return ans


def partition(s: str):
    def is_palindrome(word):
        return word == word[-1::-1]

    def backtrack(current_path, s):

        if len(s) == 0:
            ans.append(current_path)
            return

        for i in range(1, len(s) + 1):

            if is_palindrome(s[:i]):
                backtrack(current_path + [s[:i]], s[i:])

    ans = []
    backtrack([], s)
    return ans


def letterCombinations(digits: str):
    number_mapping = {'2': 'abc', '3': 'def',
                      '4': 'ghi', '5': 'jkl', '6': 'mno',
                      '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}

    def backtrack(current_path, index):

        if index == len(digits):
            ans.append(current_path)
            return

        for letter in number_mapping[digits[index]]:
            backtrack(current_path + letter, index + 1)

    ans = []
    if len(digits) == 0:
        return []
    backtrack('', 0)
    return ans
