# Linked List Cycle I（环形链表）
# 英文版：https://leetcode.com/problems/linked-list-cycle/
# 中文版：https://leetcode-cn.com/problems/linked-list-cycle/

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        r, n, p = [], lists and lists.pop(), None
        while lists or n: r[len(r):], n = ([n], n.next or lists and lists.pop()) if n else ([], lists.pop())
        for n in sorted(r, key=lambda x: x.val, reverse=True): n.next, p = p, n
        return n if r else []

# Merge k Sorted Lists（合并 k 个排序链表）
# 英文版：https://leetcode.com/problems/merge-k-sorted-lists/
# 中文版：https://leetcode-cn.com/problems/merge-k-sorted-lists/

class Solution(object):
    def hasCycle(self, head):
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                return True
        return False