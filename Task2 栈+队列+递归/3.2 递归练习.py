# Climbing Stairs（爬楼梯）
# 英文版：https://leetcode.com/problems/climbing-stairs/
# 中文版：https://leetcode-cn.com/problems/climbing-stairs/

class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n ==1: return 1
        elif n ==2: return 2
        else:
            a = 1
            b = 2
            for i in range(n-2):
                a,b = b,a+b

            return b