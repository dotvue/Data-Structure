# 实现一个二叉查找树，并且支持插入、删除、查找操作

class Node:

    def __init__(self, data):

        self.data = data

        self.lchild = None

        self.rchild = None

 

class BST:

    def __init__(self, node_list):

        self.root = Node(node_list[0])

        for data in node_list[1:]:

            self.insert(data)

 

    # 查找

    def search(self, node, parent, data):

        if node is None:

            return False, node, parent

        if node.data == data:

            return True, node, parent

        if node.data > data:

            return self.search(node.lchild, node, data)

        else:

            return self.search(node.rchild, node, data)

 

    # 插入

    def insert(self, data):

        flag, n, p = self.search(self.root, self.root, data)

        if not flag:

            new_node = Node(data)

            if data > p.data:

                p.rchild = new_node

            else:

                p.lchild = new_node

 

    # 删除

    def delete(self, root, data):

        flag, n, p = self.search(root, root, data)

        if flag is False:

            print "无该关键字，删除失败"

        else:

            if n.lchild is None:

                if n == p.lchild:

                    p.lchild = n.rchild

                else:

                    p.rchild = n.rchild

                del p

            elif n.rchild is None:

                if n == p.lchild:

                    p.lchild = n.lchild

                else:

                    p.rchild = n.lchild

                del p

            else:  # 左右子树均不为空

                pre = n.rchild

                if pre.lchild is None:

                    n.data = pre.data

                    n.rchild = pre.rchild

                    del pre

                else:

                    next = pre.lchild

                    while next.lchild is not None:

                        pre = next

                        next = next.lchild

                    n.data = next.data

                    pre.lchild = next.rchild

                    del p

# 实现查找二叉查找树中某个节点的后继、前驱节点

class Solution:
    def GetNext(self, pNode):
        # write code here
        dummy = pNode
        #找到根节点
        while dummy.next:
            dummy = dummy.next
        #中序遍历
        ls = self.inorderTraversal(dummy)
        #找pNode所在索引的下一个
        if ls.index(pNode) != (len(ls)-1):
            return ls[ls.index(pNode)+1]
        else:
            return None

    def GetBefore(self, pNode):
        # write code here
        dummy = pNode
        #找到根节点
        while dummy.next:
            dummy = dummy.next
        #中序遍历
        ls = self.inorderTraversal(dummy)
        #找pNode所在索引的下一个
        if ls.index(pNode) != 0:
            return ls[ls.index(pNode)-1]
        else:
            return None

    def inorderTraversal(self, root):
        if root==None:
            return []
        return self.inorderTraversal(root.left)+[root]+self.inorderTraversal(root.right)

# 实现二叉树前、中、后序以及按层遍历

class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if root==None:
            return []
        return [root.val]+self.preorderTraversal(root.left)+self.preorderTraversal(root.right)

    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root==None:
            return []
        return self.inorderTraversal(root.left)+[root.val]+self.inorderTraversal(root.right)

    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if root==None:
            return []
        return self.postorderTraversal(root.left)+self.postorderTraversal(root.right)+[root.val]
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        result = []
        queue = collections.deque()
        queue.append(root)
        while queue:
            level_size = len(queue)
            current_level = []
            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            result.append(current_level)
        return result