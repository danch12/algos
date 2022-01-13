def search(nums, target: int):
    def b_search(nums, target, right, left):
        if left > right:
            return -1

        mid = left + (right - left) // 2

        if target == nums[mid]:
            return mid

        if target < nums[mid]:
            return b_search(nums, target, mid - 1, left)

        if target > nums[mid]:
            return b_search(nums, target, right, mid + 1)

    return b_search(nums, target, len(nums) - 1, 0)


def peakIndexInMountainArray(arr) -> int:
    def b_search(arr, left, right):
        if left > right:
            return -1
        mid = left + (right - left) // 2
        if arr[mid - 1] < arr[mid] and arr[mid + 1] < arr[mid]:
            return mid

        if arr[mid - 1] > arr[mid]:
            return b_search(arr, left, mid)

        if arr[mid + 1] > arr[mid]:
            return b_search(arr, mid + 1, right)

    return b_search(arr, 0, len(arr) - 1)


# find min in rotated sorted array
def findMin(nums) -> int:
    def b_search(compare_num, nums, left, right):
        if len(nums) == 1:
            return nums[0]

        if nums[0] < nums[right]:
            return nums[0]

        mid = left + (right - left) // 2

        if nums[mid - 1] > nums[mid]:
            return nums[mid]

        if nums[mid] > nums[mid + 1]:
            return nums[mid + 1]

        if nums[mid] > compare_num:
            return b_search(compare_num, nums, mid + 1, right)

        if nums[mid] < compare_num:
            return b_search(compare_num, nums, left, mid - 1)

    return b_search(nums[0], nums, 0, len(nums) - 1)


# multiple peaks
def findPeakElement(nums) -> int:
    def b_search(arr, left, right):
        if left == right:
            return left
        mid = (left + right) // 2

        if nums[mid] > nums[mid + 1]:
            return b_search(arr, left, mid)

        if nums[mid] < nums[mid + 1]:
            return b_search(arr, mid + 1, right)

    return b_search(nums, 0, len(nums) - 1)


def search_rotated(nums, target: int) -> int:
    def b_search(nums, target, left, right):
        if left > right:
            return -1

        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid

        # inflection point to the right. Left is strictly increasing

        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                return b_search(nums, target, left, mid - 1)
            else:
                return b_search(nums, target, mid + 1, right)
        # inflection point to the left.  right is strictly increasing
        else:
            if nums[right] >= target > nums[mid]:
                return b_search(nums, target, mid + 1, right)
            else:
                return b_search(nums, target, left, mid - 1)

    return b_search(nums, target, 0, len(nums) - 1)