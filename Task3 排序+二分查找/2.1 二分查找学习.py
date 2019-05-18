# 实现一个有序数组的二分查找算法

def binarySearch(alist, item):
        first = 0
        last = len(alist)-1
        found = False

        while first<=last and not found:
            midpoint = (first + last)//2
            if alist[midpoint] == item:
                found = True
            else:
                if item < alist[midpoint]:
                    last = midpoint-1
                else:
                    first = midpoint+1

        return found

# 实现模糊二分查找算法（比如大于等于给定值的第一个元素）

def fuzzybinarySearch(alist, item):
        first = 0
        last = len(alist)-1
        found = False

        while first<=last and not found:
            midpoint = (first + last)//2
            if alist[midpoint] == item or (alist[midpoint]>item and alist[midpoint-1] <item):
                found = True
                position=midpoint
            elif alist[first] > item:
                found = True
                position=first
            else:
                if item < alist[midpoint]:
                    last = midpoint-1
                else:
                    first = midpoint+1

        return position