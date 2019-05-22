# Path Sum（路径总和）
# 英文版：https://leetcode.com/problems/path-sum/
# 中文版：https://leetcode-cn.com/problems/path-sum/

class Solution(object):
    def hasPathSum(self, root, sum):
        if root == None:
            return False
        if root.left == None and root.right == None:
            return root.val == sum
        if root.left == None:
            return self.hasPathSum(root.right,sum - root.val)
        if root.right == None:
            return self.hasPathSum(root.left,sum - root.val)
        return self.hasPathSum(root.left,sum - root.val) or self.hasPathSum(root.right,sum - root.val)