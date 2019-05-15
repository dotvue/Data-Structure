# 用数组实现一个顺序队列

class Stack(object):
    def __init__(self):
        self.stack = []

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        return self.stack.pop(0)

    def peek(self):
        if self.is_empty():
            return None

        return self.stack[-1]

    def is_empty(self):
        return not self.stack

# 用链表实现一个链式队列

class Node(object):
    def __init__(self, value):
        self.value = value
        self.next = None


class Stack(object):

    def __init__(self):
        self.begin = None
        self.stack = None

    def push(self, value):
        node = Node(value)
        if self.is_empty():
            self.begin = node
            self.stack = node
        else:
            self.stack.next = node
            self.stack = node

    def pop(self):
        node = self.begin
        self.begin = node.next
        return node.value

    def peek(self):
        return self.stack.value

    def is_empty(self):
        return not self.stack

# 实现一个循环队列