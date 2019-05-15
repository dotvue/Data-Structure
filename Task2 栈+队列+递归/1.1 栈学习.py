# 用数组实现一个顺序栈

class Stack(object):
    def __init__(self):
        self.stack = []

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        return self.stack.pop()

    def peek(self):
        if self.is_empty():
            return None

        return self.stack[-1]

    def is_empty(self):
        return not self.stack

# 用链表实现一个链式栈

class Node(object):
    def __init__(self, value):
        self.value = value
        self.pre = None


class Stack(object):

    def __init__(self):
        self.stack = None

    def push(self, value):
        node = Node(value)
        if self.is_empty():
            self.stack = node
        else:
            node.pre = self.stack
            self.stack = node

    def pop(self):
        node = self.stack
        self.stack = node.pre
        return node.value

    def peek(self):
        return self.stack.value

    def is_empty(self):
        return not self.stack

# 编程模拟实现一个浏览器的前进、后退功能

