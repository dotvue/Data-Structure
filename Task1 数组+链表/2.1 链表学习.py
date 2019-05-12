# 实现单链表、循环链表、双向链表，支持增删操作
class Node(object):

    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleLinkList(object):
    """单链表"""

    def __init__(self, node=None):
        self.__head = node

    def is_empty(self):
        return self.__head is None

    def length(self):
        cur = self.__head
        count = 0
        while cur is not None:
            count += 1
            cur = cur.next
        return count

    def add(self, item):
        # 头部插入
        node = Node(item)
        node.next = self.__head
        self.__head = node

    def append(self, item):
        # 尾部插入
        node = Node(item)
        if self.is_empty():
            self.__head = node
        else:
            cur = self.__head
            while cur.next is not None:
                cur = cur.next
            cur.next = node

    def insert(self, pos, item):
        if pos <= 0:
            self.add(item)
        elif pos > (self.length() - 1):
            self.append(item)
        else:
            node = Node(item)
            pre = self.__head
            count = 0
            while count < (pos - 1):
                count += 1
                pre = pre.next
            node.next = pre.next
            pre.next = node

    def delete(self, item):
        pre = self.__head
        if pre is None:
            return False
        elif pre.elem == item:
            self.__head = pre.next
            return True
        else:
            while pre.next is not None:
                if pre.next.elem == item:
                    pre.next = pre.next.next
                    return True
                else:
                    pre = pre.next
            return False

class Node(object):

    def __init__(self, item):
        self.elem = item
        self.next = None
        self.prev = None


class DoubleLinkList(object):
    """双向链表"""

    def __init__(self, node=None):
        self.__head = node

    def is_empty(self):
        return self.__head is None

    def length(self):
        cur = self.__head
        count = 0
        while cur is not None:
            count += 1
            cur = cur.next
        return count

    def append(self, item):
        node = Node(item)
        if self.is_empty():
            self.__head = node
        else:
            cur = self.__head
            while cur.next is not None:
                cur = cur.next
            cur.next = node
            node.prev = cur

    def add(self, item):
        node = Node(item)
        node.next = self.__head
        self.__head = node
        node.next.prev = node

    def insert(self, pos, item):
        if pos <= 0:
            self.add(item)
        elif pos > (self.length() - 1):
            self.append(item)
        else:
            cur = self.__head
            counter = 0
            while counter is not pos:
                counter += 1
                cur = cur.next
            node = Node(item)
            node.prev = cur.prev
            cur.prev.next = node
            node.next = cur
            cur.prev = node

    def delete(self, item):
        cur = self.__head
        while cur is not None:
            if cur.elem == item:
                if cur == self.__head:
                    self.__head = cur.next
                    if cur.next is not None:
                        cur.next.prev = None
                else:
                    cur.prev.next = cur.next
                    if cur.next is not None:
                        cur.next.prev = cur.prev
                return True
            else:
                cur = cur.next
        return False

class Node(object):

    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleLinkList(object):
    """单向循环链表"""

    def __init__(self, node=None):
        self.__head = node
        if node is not None:
            node.next = node

    def is_empty(self):
        return self.__head is None

    def length(self):
        if self.is_empty():
            return 0
        cur = self.__head
        count = 1
        while cur.next != self.__head:
            count += 1
            cur = cur.next
        return count

    def add(self, item):
        node = Node(item)
        if self.is_empty():
            self.__head = node
            node.next = self.__head

        else:
            cur = self.__head
            while cur.next != self.__head:
                cur = cur.next
            node.next = self.__head
            self.__head = node
            cur.next = node

    def append(self, item):
        node = Node(item)
        if self.is_empty():
            self.__head = node
            node.next = node
        else:
            cur = self.__head
            while cur.next != self.__head:
                cur = cur.next
            cur.next = node
            node.next = self.__head

    def insert(self, pos, item):
        if pos <= 0:
            self.add(item)
        elif pos > (self.length() - 1):
            self.append(item)
        else:
            node = Node(item)
            pre = self.__head
            count = 0
            while count < (pos - 1):
                count += 1
                pre = pre.next
            node.next = pre.next
            pre.next = node

    def delete(self, item):
        pre = self.__head
        cur = self.__head
        if cur is None:
            return False
        elif cur.elem == item:
            if cur.next == self.__head:
                self.__head = None
                return True
            else:
                while pre.next != self.__head:
                    pre = pre.next
                self.__head = cur.next
                pre.next = self.__head
                return True
        else:
            while pre.next != self.__head:
                if pre.next.elem == item:
                    pre.next = pre.next.next
                    return True
                else:
                    pre = pre.next
            return False

# 实现单链表反转

def rev(SingleLinkList):
    # 单链表的反转
    pre = SingleLinkList
    cur = SingleLinkList.next
    pre.next = None
    while cur:
        temp = cur.next
        cur.next = pre
        pre = cur
        cur = temp
    return pre

# 实现两个有序的链表合并为一个有序链表

# 实现求链表的中间结点
