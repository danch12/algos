def containsDuplicate(nums):
    num_dict = {}
    for num in nums:
        if num not in num_dict:
            num_dict[num] = 1
        elif num_dict[num] >= 1:
            return True
    return False


def containsDuplicate2(nums):
    return len(nums) != len(set(nums))


def productExceptSelf(nums):
    ans = [1] * len(nums)
    prefix = 1
    postfix = 1
    for i in range(len(nums)):
        ans[i] *= prefix  # prefix product from one end
        prefix *= nums[i]
        ans[-1 - i] *= postfix  # suffix product from other end
        postfix *= nums[-1 - i]
    return ans


def setZeroes(matrix):
    """
    Do not return anything, modify matrix in-place instead.
    """
    h, w = len(matrix), len(matrix[0])
    is_col = False
    for i in range(h):
        if matrix[i][0] == 0:
            is_col = True
        for j in range(1, w):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0

    for i in range(1, h):
        for j in range(1, w):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0

    if matrix[0][0] == 0:
        for j in range(w):
            matrix[0][j] = 0

    if is_col:
        for i in range(h):
            matrix[i][0] = 0


def spiralOrder(matrix):
    h_upper, w_upper = len(matrix), len(matrix[0])
    h_lower, w_lower = 1, 0
    i = 0
    j = 0
    right = True
    down = True
    turn = 'h'
    ans = []
    travelled = 0
    while travelled != len(matrix) * len(matrix[0]):
        ans.append(matrix[i][j])
        travelled += 1

        if right and turn == 'h' and j == w_upper - 1:
            right = False
            turn = 'v'
            w_upper -= 1

        if not right and turn == 'h' and j == w_lower:
            right = True
            turn = 'v'
            w_lower += 1

        if down and turn == 'v' and i == h_upper - 1:
            down = False
            turn = 'h'
            h_upper -= 1

        if not down and turn == 'v' and i == h_lower:
            down = True
            turn = 'h'
            h_lower += 1

        if right and turn == 'h':
            j += 1

        if not right and turn == 'h':
            j -= 1

        if down and turn == 'v':
            i += 1

        if not down and turn == 'v':
            i -= 1
    return ans


def rotate90(matrix):
    """
    Do not return anything, modify matrix in-place instead.
    """

    def transpose(matrix, h, w):

        for i in range(h):
            for j in range(i, w):
                temp = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = temp

    def reverse(matrix, h):
        for i in range(h):
            matrix[i] = matrix[i][::-1]

    h, w = len(matrix), len(matrix[0])
    transpose(matrix, h, w)
    reverse(matrix, h)


# longest consequtive streak in unordered list
def longestConsecutive(nums):
    longest_streak = 0
    num_set = set(nums)

    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1

            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1

            longest_streak = max(longest_streak, current_streak)

    return longest_streak


