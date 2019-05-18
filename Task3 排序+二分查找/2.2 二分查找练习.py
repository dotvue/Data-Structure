# Sqrt(x) （x 的平方根）
# 英文版：https://leetcode.com/problems/sqrtx/
# 中文版：https://leetcode-cn.com/problems/sqrtx/

class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x == 0:
            return 0
        left = 1
        right = x / 2 + 1
   while left <= right:
        mid = left + (right - left) / 2
        sq = x / mid
        if sq > mid:
            left = mid + 1
        elif sq < mid:
            right = mid - 1
        else:
            return mid
    return right