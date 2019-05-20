# Reverse String （反转字符串）
# 英文版：https://leetcode.com/problems/reverse-string/
# 中文版：https://leetcode-cn.com/problems/reverse-string/

class Solution:
    def reverseString(self, s: List[str]) -> None:
        for i in range(len(s)//2):
            s[i],s[len(s)-i-1]=s[len(s)-i-1],s[i]

# Reverse Words in a String（翻转字符串里的单词）
# 英文版：https://leetcode.com/problems/reverse-words-in-a-string/
# 中文版：https://leetcode-cn.com/problems/reverse-words-in-a-string/

class Solution:
    def reverseWords(self, s: str) -> str:
        if s == '':
            return s
        ls = s.split()
        if ls == []:
            return ''
        result = ''
        for i in range(0,len(ls)-1):
            result += ls[len(ls)-1-i]+' '
        result += ls[0]
        return result

# String to Integer (atoi)（字符串转换整数 (atoi)）[作为可选]
# 英文版：https://leetcode.com/problems/string-to-integer-atoi/
# 中文版：https://leetcode-cn.com/problems/string-to-integer-atoi/

class Solution:
    def myAtoi(self, str: str) -> int:
        #strip()去除空格
        stripS=str.strip()
        if stripS == '' or stripS == '-' or stripS == '+':
            return 0
        s1 = re.match('[^\d]+', (stripS.lstrip('-')).lstrip('+'))
        if s1!=None:
            return 0
        else:
            s1 = re.search('\-*\+*\d+', stripS).group()
        if s1[:2] == '--' or s1[:2]=='-+' or s1[:2]=='++':
            return 0
        result = int(s1)
        if result >0:
            return 2147483647 if result>2147483647 else result
        else:
            return -2147483648 if result<-2147483648 else result