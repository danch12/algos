class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def hasCycle(head) -> bool:
    if not head:
        return False
    fast = head.next
    slow = head

    while fast is not None and fast.next is not None:
        if fast == slow:
            return True

        fast = fast.next.next
        slow = slow.next

    return False


def middleNode(head):
    slow = head
    fast = head

    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next

    return slow


def isPalindrome(head):
    mid = middleNode(head)
    traversal = mid
    stack = []

    while traversal is not None:
        stack.append(traversal.val)
        traversal = traversal.next

    while len(stack) > 0:
        comp = stack.pop()
        if head.val != comp:
            return False
        head = head.next
    return True


def reverseList(head):
    if not head:
        return
    start = head.next
    follower = head
    follower.next = None
    while start:
        temp = start
        start = start.next
        temp.next = follower
        follower = temp

    return follower


# this is just two pointers
def mergeTwoLists(list1, list2):
    ans = ListNode(None)
    temp = ans
    while list1 and list2:
        if list1.val < list2.val:
            temp.next = ListNode(list1.val)
            list1 = list1.next
        else:
            temp.next = ListNode(list2.val)
            list2 = list2.next

        temp = temp.next

    if list1:
        temp.next = list1

    if list2:
        temp.next = list2

    return ans.next


def detect_cycle_pos(head):
    if head is None:
        return
    slow = head
    fast = head
    while fast is not None:
        fast = fast.next
        if fast is None:
            return None
        fast = fast.next

        slow = slow.next
        if fast == slow:
            break

    if fast is None:
        return None
    while slow != head:
        slow = slow.next
        head = head.next

    return head


def addTwoNumbers(l1, l2):
    carry = 0
    head = ListNode(None)
    ans = head
    while l1 is not None and l2 is not None:
        sum_val = l1.val + l2.val + carry
        if sum_val >= 10:
            carry = 1
            sum_val = sum_val % 10
        else:
            carry = 0
        head.next = ListNode(sum_val)
        head = head.next
        l1 = l1.next
        l2 = l2.next

    while l1:
        sum_val = l1.val + carry
        if sum_val >= 10:
            carry = 1
            sum_val = sum_val % 10
        else:
            carry = 0
        head.next = ListNode(sum_val)
        head = head.next
        l1 = l1.next

    while l2:
        sum_val = l2.val + carry
        if sum_val >= 10:
            carry = 1
            sum_val = sum_val % 10
        else:
            carry = 0
        head.next = ListNode(sum_val)
        head = head.next
        l2 = l2.next

    if carry == 1:
        head.next = ListNode(1)

    return ans.next


def removeNthFromEnd(head, n):
    fast = slow = head
    for _ in range(n):
        fast = fast.next
    if not fast:
        return head.next
    while fast.next:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return head


def sortList(head):
    def find_mid(head):
        slow = fast = head

        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        return slow

    def merge(left, right):

        head = ListNode(None)
        traverser = head
        while left and right:
            if left.val > right.val:
                traverser.next = right
                right = right.next
            else:
                traverser.next = left
                left = left.next
            traverser = traverser.next

        while left:
            traverser.next = left
            left = left.next
            traverser = traverser.next

        while right:
            traverser.next = right
            right = right.next
            traverser = traverser.next

        return head.next

    def m_sort(head):
        if not head or not head.next:
            return head

        mid = find_mid(head)
        right = mid.next
        mid.next = None

        left = m_sort(head)
        right = m_sort(right)
        head = merge(left, right)
        return head

    return m_sort(head)


def reorderList(head):
    """
    Do not return anything, modify head in-place instead.
    """

    def find_mid(head):
        slow = fast = head

        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        return slow

    def reverseList(head):
        if not head:
            return
        start = head.next
        follower = head
        follower.next = None
        while start:
            temp = start
            start = start.next
            temp.next = follower
            follower = temp

        return follower

    mid = find_mid(head)

    right = mid.next
    mid.next = None

    mid = reverseList(right)

    while mid:
        temp_head = head
        temp_mid = mid
        head = head.next
        mid = mid.next
        temp_head.next = temp_mid
        temp_mid.next = head
