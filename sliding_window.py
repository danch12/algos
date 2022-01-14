def minSubArrayLen(target: int, nums) -> int:
    min_len = float('inf')

    slow_pointer = 0
    current_sum = 0
    for fast_pointer in range(0, len(nums)):
        current_sum += nums[fast_pointer]

        while current_sum >= target:
            if current_sum >= target:
                if fast_pointer - slow_pointer < min_len:
                    min_len = fast_pointer - slow_pointer + 1

            current_sum -= nums[slow_pointer]
            slow_pointer += 1

    if min_len == float('inf'):
        return 0
    return min_len


def totalFruit(fruits) -> int:
    largest_basket = 0

    current_fruit_count = {}
    num_fruits = 0
    current_basket = 0
    slow_pointer = 0
    for fast_pointer in range(0, len(fruits)):

        if fruits[fast_pointer] in current_fruit_count:
            current_fruit_count[fruits[fast_pointer]] += 1
            current_basket += 1
        else:
            current_fruit_count[fruits[fast_pointer]] = 1
            num_fruits += 1
            current_basket += 1
        while num_fruits > 2:
            if fruits[slow_pointer] in current_fruit_count:
                current_fruit_count[fruits[slow_pointer]] -= 1
                current_basket -= 1
                if current_fruit_count[fruits[slow_pointer]] == 0:
                    del current_fruit_count[fruits[slow_pointer]]
                    num_fruits -= 1
            slow_pointer += 1

        if current_basket > largest_basket:
            largest_basket = current_basket

    return largest_basket


# Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise.
# In other words, return true if one of s1's permutations is the substring of s2.
def checkInclusion(s1: str, s2: str) -> bool:
    letters_used = {}

    for char in s1:
        letters_used[char] = letters_used.get(char, 0) + 1

    letters_s2 = {}
    slow_pointer = 0
    for fast_pointer in range(0, len(s2)):
        letters_s2[s2[fast_pointer]] = letters_s2.get(s2[fast_pointer], 0) + 1
        count = 0
        for key in letters_used:
            if letters_s2.get(key, 0) == letters_used[key]:
                count += 1
        if count == len(letters_used):
            return True

        if s2[fast_pointer] not in letters_used:

            letters_s2 = {}
            slow_pointer = fast_pointer + 1
        else:
            while letters_s2[s2[fast_pointer]] > letters_used.get(s2[fast_pointer], 0):
                letters_s2[s2[slow_pointer]] -= 1
                slow_pointer += 1

    return False


def characterReplacement(s: str, k: int) -> int:
    letters = {}
    most_frequent_letter = 0

    slow_pointer = 0
    longest = 0
    for fast_pointer in range(len(s)):
        letters[s[fast_pointer]] = letters.get(s[fast_pointer], 0) + 1
        most_frequent_letter = max(most_frequent_letter, letters[s[fast_pointer]])
        # letters to change = len of string - most_frequent
        letters_to_change = (fast_pointer - slow_pointer + 1) - most_frequent_letter

        if letters_to_change > k:
            letters[s[slow_pointer]] -= 1
            slow_pointer += 1
        longest = max(longest, fast_pointer - slow_pointer + 1)

    return longest


# Given a string s, find the length of the longest substring without repeating characters.
def lengthOfLongestSubstring(self, s: str) -> int:
    max_length = 0
    letters = {}
    current_longest = 0
    slow_pointer = 0
    for fast_pointer in range(len(s)):

        letters[s[fast_pointer]] = letters.get(s[fast_pointer], 0) + 1

        current_longest += 1

        while letters[s[fast_pointer]] > 1:
            letters[s[slow_pointer]] -= 1

            slow_pointer += 1
            current_longest -= 1

        max_length = max(max_length, current_longest)

    return max_length
