# 实现一个字符集，只包含 a～z 这 26 个英文字母的 Trie 树

class Trie:
    # word_end = -1
 
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}
        self.word_end = -1
 
    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        curNode = self.root
        for c in word:
            if not c in curNode:
                curNode[c] = {}
            curNode = curNode[c]
          
        curNode[self.word_end] = True
 
    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        curNode = self.root
        for c in word:
            if not c in curNode:
                return False
            curNode = curNode[c]
            
        # Doesn't end here
        if self.word_end not in curNode:
            return False
        
        return True
 
    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        curNode = self.root
        for c in prefix:
            if not c in curNode:
                return False
            curNode = curNode[c]
        
        return True

# 实现朴素的字符串匹配算法

def naive_match(s, p):
    m = len(s); n = len(p)
    for i in range(m-n+1):#起始指针i
        if s[i:i+n] == p:
            return True
    return False