# 实现一个小顶堆、大顶堆、优先级队列

class ZHeap:
    def __init__(self, item=[]):
        # 初始化。item为数组
        self.items = item
        self.heapsize = len(self.items)

    def LEFT(self, i):
        return 2 * i + 1

    def RIGHT(self, i):
        return 2 * i + 2

    def PARENT(self, i):
        return (i - 1) / 2

    def MIN_HEAPIFY(self, i):
        # 最小堆化：使以i为根的子树成为最小堆
        l = self.LEFT(i)
        r = self.RIGHT(i)
        if l < self.heapsize and self.items[l] < self.items[i]:
            smallest = l
        else:
            smallest = i

        if r < self.heapsize and self.items[r] < self.items[smallest]:
            smallest = r

        if smallest != i:
            self.items[i], self.items[smallest] = self.items[smallest], self.items[i]
            self.MIN_HEAPIFY(smallest)

    def INSERT(self, val):
        # 插入一个值val，并且调整使满足堆结构
        self.items.append(val)
        idx = len(self.items) - 1
        parIdx = self.PARENT(idx)
        while parIdx >= 0:
            if self.items[parIdx] > self.items[idx]:
                self.items[parIdx], self.items[idx] = self.items[idx], self.items[parIdx]
                idx = parIdx
                parIdx = self.PARENT(parIdx)
            else:
                break
        self.heapsize += 1

    def DELETE(self):
        last = len(self.items) - 1
        if last < 0:
            # 堆为空
            return None
        # else:
        self.items[0], self.items[last] = self.items[last], self.items[0]
        val = self.items.pop()
        self.heapsize -= 1
        self.MIN_HEAPIFY(0)
        return val


    def BUILD_MIN_HEAP(self):
        # 建立最小堆, O(nlog(n))
        i = self.PARENT(len(self.items) - 1)
        while i >= 0:
            self.MIN_HEAPIFY(i)
            i -= 1

    def SHOW(self):
        print self.items


class ZPriorityQ(ZHeap):
    def __init__(self, item=[]):
        ZHeap.__init__(self, item)

    def enQ(self, val):
        ZHeap.INSERT(self, val)

    def deQ(self):
        val = ZHeap.DELETE(self)
        return val

class MaxHeap:
    def __init__(self, data):
        data.insert(0, None)
        self.heap = data
        self.heapSize = 0
        for i in range(1,len(self.heap)):
            self.heapSize += 1
            self.__bubble(i)

    def __sink(self, pos):
        left, right = 2*pos, 2*pos+1
        next = pos
        if left <= self.heapSize and self.compare(self.heap[left], self.heap[next]) > 0:
            next = left
        if right <= self.heapSize and self.compare(self.heap[right], self.heap[next]) > 0:
            next = right
        if next == pos:
            return
        self.__exchange(pos, next)
        return self.__sink(next)

    def __bubble(self, pos): # build
        if pos <= 1:
            return
        ppos = pos/2
        if self.compare(self.heap[pos], self.heap[ppos]) > 0:
            self.__exchange(pos, ppos)
            return self.__bubble(ppos)

    def compare(self, a, b):
        return a - b

    def __exchange(self, i, j):
        temp = self.heap[i]
        self.heap[i] = self.heap[j]
        self.heap[j] = temp

    def sort(self):
        while self.heapSize > 1:
            self.__exchange(1, self.heapSize)
            self.heapSize -= 1
            self.__sink(1)
        self.heap.remove(None)
        return self.heap

# 实现堆排序

def sift_down(arr, start, end):

    root = start

    while True:

        # 从root开始对最大堆调整

        child = 2 * root + 1

        if child > end:

            break

 

        # 找出两个child中交大的一个

        if child + 1 <= end and arr[child] < arr[child + 1]:

            child += 1

 

        if arr[root] < arr[child]:

            # 最大堆小于较大的child, 交换顺序

            arr[root], arr[child] = arr[child], arr[root]

 

            # 正在调整的节点设置为root

            root = child

        else:

            # 无需调整的时候, 退出

            break
 

def heap_sort(arr):

    # 从最后一个有子节点的孩子还是调整最大堆

    first = len(arr) // 2 - 1

    for start in range(first, -1, -1):

        sift_down(arr, start, len(arr) - 1)

 

    # 将最大的放到堆的最后一个, 堆-1, 继续调整排序

    for end in range(len(arr) -1, 0, -1):

        arr[0], arr[end] = arr[end], arr[0]

        sift_down(arr, 0, end - 1)

# 利用优先级队列合并 K 个有序数组

class Solution:
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """

        heap = []
        for ln in lists:
            if ln:
                heap.append((ln.val, ln))
        dummy = ListNode(0)
        cur = dummy
        heapq.heapify(heap)
        while heap:
            valu, ln_index = heapq.heappop(heap)
            cur.next = ln_index
            cur = cur.next
            if ln_index.next:
                heapq.heappush(heap, (ln_index.next.val, ln_index.next))
        return dummy.next

# 求一组动态数据集合的最大 Top K

def findKthLargest(self, nums, k):
        import heapq
        return heapq.nlargest(k, nums)[-1]