# Design Circular Deque（设计一个双端队列）
# 英文版：https://leetcode.com/problems/design-circular-deque/
# 中文版：https://leetcode-cn.com/problems/design-circular-deque/

class MyCircularDeque(object):

    def __init__(self, k):
        """
        Initialize your data structure here. Set the size of the deque to be k.
        :type k: int
        """
        from collections import deque
        self.cdeque = deque([])
        self.cur = 0
        self.k = k
        

    def insertFront(self, value):
        """
        Adds an item at the front of Deque. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        if self.cur + 1 <= self.k:
            self.cdeque.appendleft(value)
            self.cur += 1
            return True
        return False
        

    def insertLast(self, value):
        """
        Adds an item at the rear of Deque. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        if self.cur + 1 <= self.k:
            self.cdeque.append(value)
            self.cur += 1
            return True
        return False

    def deleteFront(self):
        """
        Deletes an item from the front of Deque. Return true if the operation is successful.
        :rtype: bool
        """
        if self.cur>=1:
            self.cdeque.popleft()
            self.cur -= 1
            return True
        return False
        

    def deleteLast(self):
        """
        Deletes an item from the rear of Deque. Return true if the operation is successful.
        :rtype: bool
        """
        if self.cur>=1:
            self.cdeque.pop()
            self.cur -= 1
            return True
        return False
        

    def getFront(self):
        """
        Get the front item from the deque.
        :rtype: int
        """
        if self.cur != 0:
            return self.cdeque[0]
        return -1
        

    def getRear(self):
        """
        Get the last item from the deque.
        :rtype: int
        """
        if self.cur != 0:
            return self.cdeque[-1]
        return -1
        

    def isEmpty(self):
        """
        Checks whether the circular deque is empty or not.
        :rtype: bool
        """
        return self.cur == 0
        

    def isFull(self):
        """
        Checks whether the circular deque is full or not.
        :rtype: bool
        """
        return self.cur == self.k

# Sliding Window Maximum（滑动窗口最大值）
# 英文版：https://leetcode.com/problems/sliding-window-maximum/
# 中文版：https://leetcode-cn.com/problems/sliding-window-maximum/

class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        if not nums or k == 0:
            return []
        if k == 1:
            return nums
        #队列里保存的是索引
        queue = [0]
        res = []
        for i in range(1, len(nums)):
            #队列的第一个索引始终为最大值索引，（当前索引-队列第一个索引）大于滑动窗口长度时，删除第一个索引
            if i-queue[0] >= k:
                queue.pop(0)
            #保证当前最大的值的索引是第一个
            while queue and nums[queue[-1]] < nums[i]:
                queue.pop(-1)
            queue.append(i)
            if i >= k-1:
                res.append(nums[queue[0]])
        return res