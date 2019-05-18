# 实现归并排序、快速排序、插入排序、冒泡排序、选择排序、堆排序（选做）

# 归并排序

def merge_sort(arr):
    n = len(arr)
    if n <= 1:
        return arr
    mid = n // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    arr = merge(left, right)
    return arr


def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(right[j])
            j += 1
        else:
            result.append(left[i])
            i += 1

    if i == len(left):
        for num in right[j:]:
            result.append(num)

    if j == len(right):
        for num in left[i:]:
            result.append(num)

    return result


array = [0, 1, 2, 3, 4, 3, 2, 1, 0, 5, 6, 7, 8, 9]
sorted_array = merge_sort(array)
print(sorted_array)

# 快速排序

def qsort(arr, start, end):
    base = arr[start]
    pl = start
    pr = end
    while pl < pr:
        while pl < pr and arr[pr] <= base:
            pr -= 1
        if pl == pr:
            break
        else:
            arr[pl], arr[pr] = arr[pr], arr[pl]

        while pl < pr and arr[pl] >= base:
            pl += 1
        if pl == pr:
            break
        else:
            arr[pl], arr[pr] = arr[pr], arr[pl]

    # now pl == pr
    if pl - 1 > start:
        qsort(arr, start, pl - 1)
    if pr + 1 < end:
        qsort(arr, pr + 1, end)

    return arr


array = [0, 1, 2, 3, 4, 3, 2, 1, 0, 5, 6, 7, 8, 9]
sorted_array = qsort(array, 0, len(array) - 1)
print(sorted_array)

# 插入排序

"""
它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。
"""
def insert_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and key > arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
            arr[j + 1] = key

    return arr


array = [0, 1, 2, 3, 4, 3, 2, 1, 0, 5, 6, 7, 8, 9]
sorted_array = insert_sort(array)
print(sorted_array)

# 冒泡排序

"""
它重复地走访过要排序的元素列，依次比较两个相邻的元素，如果他们的顺序（如从大到小、首字母从A到Z）错误就把他们交换过来。
走访元素的工作是重复地进行直到没有相邻元素需要交换，也就是说该元素列已经排序完成。
"""
def bubble_sort(arr):
    n = len(arr)
    # 一共进行几轮列表比较,一共是(n - 1)轮
    for i in range(n - 1):
        # 每一轮的比较,注意range的变化,这里需要进行n - 1 - i的比较,注意-i的意义(可以减少比较已经排好序的元素)
        for j in range(n - 1 - i):
            if arr[j] < arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    return arr


array = [0, 1, 2, 3, 4, 3, 2, 1, 0, 5, 6, 7, 8, 9]
sorted_array = bubble_sort(array)
print(sorted_array)

# 选择排序

"""
基本思想：第1趟，在待排序记录r1 ~ r[n]中选出最小的记录，将它与r1交换；第2趟，在待排序记录r2 ~ r[n]中选出最小的记录，将它与r2交换；
以此类推，第i趟在待排序记录r[i] ~ r[n]中选出最小的记录，将它与r[i]交换，使有序序列不断增长直到全部排序完毕。
"""
def select_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        max_index = i
        for j in range(i + 1, n):
            if arr[max_index] < arr[j]:
                max_index = j
        arr[max_index], arr[i] = arr[i], arr[max_index]

    return arr


array = [0, 1, 2, 3, 4, 3, 2, 1, 0, 5, 6, 7, 8, 9]
sorted_array = select_sort(array)
print(sorted_array)

# 堆排序

"""
堆排序的基本思想是：将待排序序列构造成一个大顶堆，此时，整个序列的最大值就是堆顶的根节点。
将其与末尾元素进行交换，此时末尾就为最大值。
然后将剩余n-1个元素重新构造成一个堆，这样会得到n个元素的次小值。
如此反复执行，便能得到一个有序序列了
"""
heap_size = 0
Left = lambda i: 2 * i + 1
Right = lambda i: 2 * i + 2


def max_heap(arr, i):
    while True:
        l, r = Left(i), Right(i)
        # 小顶堆
        smallest = l if l < heap_size and arr[i] > arr[l] else i
        smallest = r if r < heap_size and arr[smallest] > arr[r] else smallest
        if i == smallest:
            break
        arr[i], arr[smallest] = arr[smallest], arr[i]
        i = smallest


def build_max_heap(arr):
    global heap_size
    heap_size = len(arr)
    # 从第一个非叶子节点开始
    for i in range(heap_size // 2 - 1, -1, -1):
        max_heap(arr, i)


def heap_sort(arr):
    global heap_size
    build_max_heap(arr)
    for i in range(len(arr) - 1, -1, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heap_size -= 1
        max_heap(arr, 0)
    return arr


array = [0, 1, 2, 3, 4, 3, 2, 1, 0, 5, 6, 7, 8, 9]
sorted_array = heap_sort(array)
print(sorted_array)

# 编程实现 O(n) 时间复杂度内找到一组数据的第 K 大元素

class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        return self.search(nums, 0, len(nums) - 1, k)

    def search(self, nums, start, end, k):
        m = self.partition(nums, start, end)
        if m == k - 1:
            return nums[m]
        elif m > k - 1:
            return self.search(nums, start, m - 1, k)
        else:
            return self.search(nums, m + 1, end, k)

    def partition(self, nums, start, end):
        base = nums[start]
        pl = start
        pr = end
        while pl < pr:
            while pl < pr and nums[pr] <= base:
                pr -= 1

            if pl == pr:
                break
            else:
                nums[pl], nums[pr] = nums[pr], nums[pl]

            while pl < pr and nums[pl] >= base:
                pl += 1

            if pl == pr:
                break
            else:
                nums[pl], nums[pr] = nums[pr], nums[pl]

        nums[pl] = base
        return pl