# 实战递归：完成Leetcode上的Letter Combinations of a Phone Number(17)及permutations(46) （保留往期第六天任务）

# Letter Combinations of a Phone Number(17)
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if digits:
            len1 = len(digits)
            num2letter = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
            set1 = num2letter[digits[0]]
            result = []
            def backtrack(level, str1, set1):
                '''if level == len1-1:
                    result.append(str1)
                    return '''
                for letter in set1:
                    str1 += letter
                    if level < len1-1:
                        backtrack(level+1, str1, num2letter[digits[level+1]])
                    if len(str1) == len1:
                        result.append(str1)
                    str1 = str1[:-1]
                return result
            return backtrack(0,'',num2letter[digits[0]])
        return []
# permutations(46)
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        def perm(nums, end):
            if end == 0:
                result.append(nums[:])
            else:
                for i in range(end+1):
                    nums[i], nums[end] = nums[end], nums[i]
                    perm(nums, end-1)
                    nums[i], nums[end] = nums[end], nums[i]
            return result
        return perm(nums, len(nums)-1)

# 实战DP：完成0-1背包问题实现(自我实现)及Leetcode上Palindrome Partitioning II(132) （保留往期第七天任务）

class Solution:
    def minCut(self, s: str) -> int:
        if not s or s == s[::-1]: return 0
        n = len(s)
        for i in range(1, n):
            if s[:i] == s[:i][::-1] and s[i:] == s[i:][::-1]:
                return 1
        dp = list(range(-1, n))
        for i in range(n):
            for k in range(0, min(n - i, i + 1)):
                if s[i + k] != s[i - k]:
                    break
                dp[i + k + 1] = min(dp[i + k + 1], dp[i - k] + 1)
            for k in range(1, min(n - i, i + 2)):    
                if s[i + k] != s[i - k + 1]:
                    break
                dp[i + k + 1] = min(dp[i + k + 1], dp[i - k + 1] + 1)
        return dp[n]

# Regular Expression Matching（正则表达式匹配）
# 英文版：https://leetcode.com/problems/regular-expression-matching/
# 中文版：https://leetcode-cn.com/problems/regular-expression-matching/

class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        #if not s or not p:
            #return False
        s_len = len(s)
        p_len = len(p)
        dp = [[False] * (p_len + 1) for _ in range(s_len + 1)]
        #print(dp)
        dp[0][0] = True
        for i in range(p_len):
            if p[i] == "*" and dp[0][i - 1]:
                dp[0][i + 1] = True
        #print(dp)
        for i in range(s_len):
            for j in range(p_len):
                if p[j] == s[i] or p[j] == ".":
                    dp[i + 1][j + 1] = dp[i][j]
                elif p[j] == "*":
                    if p[j - 1] != s[i]:
                        dp[i + 1][j + 1] = dp[i + 1][j - 1]
                    if p[j-1] == s[i] or p[j-1] == ".":
                        dp[i+1][j+1] = (dp[i][j+1] or dp[i+1][j]   or  dp[i+1][j-1])
        #print(dp)
        return dp[-1][-1]

# Minimum Path Sum（最小路径和）
# 英文版：https://leetcode.com/problems/minimum-path-sum/
# 中文版：https://leetcode-cn.com/problems/minimum-path-sum/

# Coin Change （零钱兑换）[作为可选]
# 英文版：https://leetcode.com/problems/coin-change/
# 中文版：https://leetcode-cn.com/problems/coin-change/

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount < 0:return -1
        dp = [float('inf')] * (amount+1)
        dp[0] = 0
        for i in range(1,amount+1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i-coin]+1)      # 状态转移方程
                    
        return dp[amount] if dp[amount] != float('inf') else -1

# Best Time to Buy and Sell Stock（买卖股票的最佳时机）[作为可选]
# 英文版：https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
# 中文版：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if (len(prices)<=1):
            return 0
        min_p=prices[0]
        max_p=0
        for i in range(len(prices)):
            min_p= min(min_p,prices[i])
            max_p= max(max_p,prices[i]-min_p)
        return max_p

# Maximum Product Subarray（乘积最大子序列）[作为可选]
# 英文版：https://leetcode.com/problems/maximum-product-subarray/
# 中文版：https://leetcode-cn.com/problems/maximum-product-subarray/

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        res = nums[0]
        mid = [0,0]
        if nums[0] >= 0:
            mid = [0,nums[0]]
        else:
            mid = [nums[0],0]
        for i in range(1,len(nums)):
            if nums[i] >= 0:
                mid[1] = max(nums[i],nums[i]*mid[1])
                mid[0] = mid[0]*nums[i]
            else:
                mid[0],mid[1] = min(nums[i],nums[i]*mid[1]), nums[i]*mid[0]
            res = max(res,mid[1])
        return res

# Triangle（三角形最小路径和）[作为可选]
# 英文版：https://leetcode.com/problems/triangle/
# 中文版：https://leetcode-cn.com/problems/triangle/

class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if len(triangle) == 0:
            return 0
        row = len(triangle) - 2
        for row in range(row, -1, -1):
            for col in range(len(triangle[row])):
                triangle[row][col] += min(triangle[row+1][col],triangle[row+1][col+1])
        return triangle[0][0]
# 原文：https://blog.csdn.net/weixin_41725746/article/details/90551455 