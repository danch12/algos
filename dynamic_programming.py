def maxProfit(prices):
    min_val = prices[0]
    best_profit = 0
    for i in range(1, len(prices)):
        if prices[i] < min_val:
            min_val = prices[i]
        elif prices[i] - min_val > best_profit:
            best_profit = prices[i] - min_val

    return best_profit


def climbStairs(n):
    def helper(n, cache={}):

        if n == 1:
            return 1

        if n == 0:
            return 0

        if n == 2:
            return 2
        if n in cache:
            return cache[n]

        result = helper(n - 1, cache) + helper(n - 2, cache)
        cache[n] = result
        return result

    return helper(n)


def maxSubArray(nums):
    largest_sum = nums[0]
    local_maximum = [nums[0]]
    for i in range(1, len(nums)):
        local_max = max(nums[i], local_maximum[i - 1] + nums[i])
        local_maximum.append(local_max)
        if local_max > largest_sum:
            largest_sum = local_max
    return largest_sum


def rob(nums):
    if len(nums) == 0:
        return 0

    if len(nums) == 1:
        return nums[0]

    if len(nums) == 2:
        return max(nums[0], nums[1])

    visited = [nums[0], max(nums[1], nums[0])]

    for i in range(2, len(nums)):
        visited.append(max(visited[i - 2] + nums[i], visited[i - 1]))

    return max(visited)


def rob_2(nums):
    def rob_helper(nums):
        if len(nums) == 0:
            return 0

        if len(nums) == 1:
            return nums[0]

        if len(nums) == 2:
            return max(nums[0], nums[1])

        visited = [nums[0], max(nums[1], nums[0])]

        for i in range(2, len(nums)):
            visited.append(max(visited[i - 2] + nums[i], visited[i - 1]))

        return max(visited)

    if len(nums) == 1:
        return nums[0]
    return max(rob_helper(nums[1:]), rob_helper(nums[:len(nums) - 1]))


def rob_different(nums):
    def rob_helper(nums):
        rob, not_rob = 0, 0

        for num in nums:
            rob, not_rob = not_rob + num, max(rob, not_rob)
        return max(rob, not_rob)

    if len(nums) == 0:
        return 0
    if len(nums) == 1:
        return nums[0]
    return max(rob_helper(nums[1:]), rob_helper(nums[:len(nums) - 1]))


def coinChange(coins, amount):
    max_val = float('inf')
    arr = [0] + [max_val] * amount
    for i in range(1, amount + 1):
        arr[i] = min([arr[i - c] if i - c >= 0 else max_val for c in coins]) + 1

    if arr[-1] == max_val:
        return -1

    return arr[-1]


def maxProduct(nums):
    curr_min, curr_max = 1, 1
    res = nums[0]
    for num in nums:
        curr_vals = (num, num * curr_min, num * curr_max)
        curr_max, curr_min = max(curr_vals), min(curr_vals)

        res = max(res, curr_max)

    return res


def lengthOfLIS(nums):
    memo = [1] * len(nums)

    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j] and memo[i] < memo[j] + 1:
                memo[i] = memo[j] + 1

    return max(memo)


def lengthOfLIS_better(nums):
    arr = [nums[0]]

    for i in range(1, len(nums)):
        if nums[i] > arr[-1]:
            arr.append(nums[i])
        else:
            for j in range(len(arr)):
                if nums[i] <= arr[j]:
                    arr[j] = nums[i]
                    break
    return len(arr)


def findNumberOfLIS(nums) -> int:
    length = [1] * len(nums)
    count = [1] * len(nums)

    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                if length[i] == length[j]:
                    length[i] = length[j] + 1
                    count[i] = count[j]
                elif length[i] == length[j] + 1:
                    count[i] += count[j]
    top_len = max(length)
    return sum([count[i] for i in range(len(count)) if length[i] == top_len])


def longestPalindrome(s: str) -> str:
    dp = [[0] * len(s) for _ in range(len(s))]
    for i in range(len(s)):
        dp[i][i] = True
    longest = 0
    substring = s[0]
    for i in range(len(s) - 1, -1, -1):
        for j in range(i + 1, len(s)):
            if j == i:
                dp[i][j] = True
            if s[j] == s[i]:
                if j - i == 1 or dp[i + 1][j - 1] == True:
                    dp[i][j] = True
            if dp[i][j] == True and j - i > longest:
                longest = j - i
                substring = s[i:j + 1]
    return substring


def wordBreak(s: str, wordDict) -> bool:
    dp = [False] * len(s)
    for i in range(len(s)):
        for word in wordDict:
            if word == s[i - len(word) + 1:i + 1]:
                if dp[i - len(word)] or i - len(word) == -1:
                    dp[i] = True

    return dp[-1]


def combinationSum4(nums, target):
    dp = [0] * (target + 1)
    dp[0] = 1
    for i in range(1, len(dp)):
        for num in nums:
            if num <= i:
                dp[i] += dp[i - num]

    return dp[-1]


def numDecodings(self, s: str) -> int:
    if s[0] == '0':
        return 0

    dp = [0] * (len(s) + 1)
    dp[1] = 1
    dp[0] = 1
    for i in range(2, len(s) + 1):

        if 0 < int(s[i - 1:i]) <= 9:
            dp[i] += dp[i - 1]

        if 10 <= int(s[i - 2:i]) <= 26:
            dp[i] += dp[i - 2]

    return dp[-1]


def uniquePaths(m: int, n: int) -> int:
    memo = [[0] * n for _ in range(m)]

    def helper(current_m, current_n, m, n):
        if current_m < 0 or current_m >= m:
            return 0

        if current_n < 0 or current_n >= n:
            return 0

        if current_n == n - 1 and current_m == m - 1:
            return 1

        if memo[current_m][current_n] > 0:
            return memo[current_m][current_n]

        right = helper(current_m, current_n + 1, m, n)
        down = helper(current_m + 1, current_n, m, n)

        memo[current_m][current_n] = right + down
        return right + down

    return helper(0, 0, m, n)


def canJump(nums) -> bool:
    current_ind = len(nums) - 1

    for i in range(len(nums) - 1, -1, -1):
        if i + nums[i] >= current_ind:
            current_ind = i

    if current_ind == 0:
        return True
    return False


# count substring palindromes
def countSubstrings(s: str) -> int:
    count = 0
    dp = [[0] * len(s) for _ in range(len(s))]
    for i in range(len(s)):
        dp[i][i] = True
        count += 1
    for i in range(len(s) - 1, -1, -1):
        for j in range(i + 1, len(s)):
            if s[i] == s[j]:
                if j - i == 1 or dp[i + 1][j - 1]:
                    dp[i][j] = True
                    count += 1
    return count


def canPartition(nums):
    total = sum(nums)
    if total % 2 != 0:
        return False
    dp = [False] * (total + 1)

    dp[0] = True

    for num in nums:
        for curr in range(total, num - 1, -1):
            # been seen before or can be reached by something that has been seen before
            dp[curr] = dp[curr] or dp[curr - num]

    return dp[total // 2]


def canPartitionKSubsets(nums, k):
    total = sum(nums)

    sub_amount = total / k

    buckets = [0] * k

    nums.sort(reverse=True)

    length = len(nums)

    def walk(i):
        if i == length:
            return len(set(buckets)) == 1
        for j in range(k):
            buckets[j] += nums[i]
            if buckets[j] <= sub_amount and walk(i + 1):
                return True
            buckets[j] -= nums[i]
            if buckets[j] == 0:
                break
        return False

    return walk(0)
