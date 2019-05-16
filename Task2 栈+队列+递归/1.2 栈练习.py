# Valid Parentheses（有效的括号）
# 英文版：https://leetcode.com/problems/valid-parentheses/
# 中文版：https://leetcode-cn.com/problems/valid-parentheses/

class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        lookup = {
            "(":")",
            "[":"]",
            "{":"}"
        }
        for alp in s:
            if alp in lookup:
                stack.append(alp)
                continue
            if stack and lookup[stack[-1]] == alp:
                stack.pop()
            else:
                return False
        return True if not stack else False

# Longest Valid Parentheses（最长有效的括号）[作为可选]
# 英文版：https://leetcode.com/problems/longest-valid-parentheses/
# 中文版：https://leetcode-cn.com/problems/longest-valid-parentheses/

class Solution:
    def longestValidParentheses(self, s: str) -> int:
        n = len(s)
        if n == 0: return 0
        dp = [0] * n
        res = 0
        for i in range(n):
            if i>0 and s[i] == ")":
                if  s[i - 1] == "(":
                    dp[i] = dp[i - 2] + 2
                elif s[i - 1] == ")" and i - dp[i - 1] - 1 >= 0 and s[i - dp[i - 1] - 1] == "(":
                    dp[i] = dp[i - 1] + 2 + dp[i - dp[i - 1] - 2]
                if dp[i] > res:
                    res = dp[i]
        return res

# Evaluate Reverse Polish Notatio（逆波兰表达式求值）
# 英文版：https://leetcode.com/problems/evaluate-reverse-polish-notation/
# 中文版：https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/

class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        f1 = lambda a,b:a+b
        f2 = lambda a,b:a-b
        f3 = lambda a,b:a*b
        f4 = lambda a,b:int(a/b)
        maps = {'+':f1,'-':f2,'*':f3,'/':f4}
        stack = []
        for i in tokens:
            if i in maps:
                a = stack.pop()
                b = stack.pop()
                stack.append(maps[i](b,a))
            else:
                i = int(i)
                stack.append(i)
        return stack[-1]