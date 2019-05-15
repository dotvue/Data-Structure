# 编程实现斐波那契数列求值 f(n)=f(n-1)+f(n-2)

class Solution:
    def fib(self, N: int) -> int:
        if N == 0:
            return 0
        if N == 1:
            return 1
        else:
            return self.fib(N-1) + self.fib(N-2)

# 编程实现求阶乘 n!

class Solution:
    def factorial(self, N: int) -> int:
        if N == 0:
            return 1
        if N == 1:
            return 1
        else:
            return N * self.factorial(N-1)

# 编程实现一组数据集合的全排列

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        if not nums:
            return []
        if len(nums) == 1:
            return [nums]
        res = []
        for x in nums:
            ys=nums[:]
            ys.remove(x)
            for y in self.permute(ys):
                res.append([x]+y)
        return res