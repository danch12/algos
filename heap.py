import heapq
import collections


def kSmallestPairs(nums1, nums2, k: int):
    if not nums1 or not nums2:
        return []

    visited = {}
    heap = []
    output = []
    visited[(0, 0)] = True
    heapq.heappush(heap, (nums1[0] + nums2[0], 0, 0))
    while len(output) < k and heap:
        value = heapq.heappop(heap)

        output.append([nums1[value[1]], nums2[value[2]]])

        # check index is in bounds and combo not been seen before
        if value[1] + 1 < len(nums1) and (value[1] + 1, value[2]) not in visited:
            total = nums1[value[1] + 1] + nums2[value[2]]
            heapq.heappush(heap, (total, value[1] + 1, value[2]))
            visited[(value[1] + 1, value[2])] = True

        if value[2] + 1 < len(nums2) and (value[1], value[2] + 1) not in visited:
            total = nums1[value[1]] + nums2[value[2] + 1]
            heapq.heappush(heap, (total, value[1], value[2] + 1))
            visited[(value[1], value[2] + 1)] = True
    return output


def frequencySort(s: str) -> str:
    frequencys = {}
    for i in range(len(s)):
        frequencys[s[i]] = frequencys.get(s[i], 0) + 1

    heap = []
    for key in frequencys.keys():
        for _ in range(frequencys[key]):
            heapq.heappush(heap, (frequencys[key], key))

    output = ''
    while len(heap) > 0:
        output = heapq.heappop(heap)[1] + output
    return output


def frequency_sort_faster(s: str) -> str:
    frequencys = collections.Counter(s)

    heap = [(freq[1], freq[0]) for freq in frequencys.items()]
    heapq.heapify(heap)

    output = ''
    while len(heap) > 0:
        value = heapq.heappop(heap)
        output = value[1] * value[0] + output
    return output


def findKthLargest(nums, k: int) -> int:
    nums = [-num for num in nums]
    heapq.heapify(nums)

    while k > 1:
        heapq.heappop(nums)
        k -= 1
    return -heapq.heappop(nums)


def reorganizeString(s: str) -> str:
    counter = collections.Counter(s)
    heap = [(-item[1], item[0]) for item in counter.items()]

    heapq.heapify(heap)
    prev_letter = ''
    prev_amount = 0
    ans = []
    while heap:
        amount, letter = heapq.heappop(heap)
        ans += [letter]
        if prev_amount < 0:
            heapq.heappush(heap, (prev_amount, prev_letter))
        amount += 1
        prev_amount = amount
        prev_letter = letter

    ans = ''.join(ans)
    if len(ans) == len(s):
        return ans
    return ''
