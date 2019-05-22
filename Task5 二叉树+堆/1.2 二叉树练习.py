# Invert Binary Tree（翻转二叉树）
# 英文版：https://leetcode.com/problems/invert-binary-tree/
# 中文版：https://leetcode-cn.com/problems/invert-binary-tree/

class Solution:
    def invertTree(self, root):
        if root:
            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)   
        return root

# Maximum Depth of Binary Tree（二叉树的最大深度）
# 英文版：https://leetcode.com/problems/maximum-depth-of-binary-tree/
# 中文版：https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/

class Solution(object):
    def maxDepth(self, root):
        if root==None:
            return 0
        return max(self.maxDepth(root.left),self.maxDepth(root.right))+1

# Validate Binary Search Tree（验证二叉查找树）[作为可选]
# 英文版：https://leetcode.com/problems/validate-binary-search-tree/
# 中文版：https://leetcode-cn.com/problems/validate-binary-search-tree/

class Solution(object):
    def isValidBST(self, root):
        if root is None:
            return True
        if root.left==None and root.right==None:
            return True 
            
        self.List=[]
        self.left_root_right(root) #调用left_root_right()函数，中序遍历二叉搜索树，将节点的值存入列表List中
        for i in range(1,len(self.List)):
            if self.List[i]<=self.List[i-1]: #通过for循环遍历列表，若当前值少于或等于前一个值，则返回False
                return False
        return True 
    
    def left_root_right(self,root):
        if root==None:
            return 
        
        self.left_root_right(root.left) #中序遍历当前子树的左子树
        self.List.append(root.val) #将当前子树的根节点的值存入列表List中
        self.left_root_right(root.right)#中序遍历当前子树的右子树