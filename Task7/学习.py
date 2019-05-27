# 通过LeetCode上【70. 爬楼梯】学习

class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        f1 = 1
        f2 = 2
        for i in range(3, n + 1):
            f1, f2 = f2, f1 + f2

        return f2

# 利用回溯算法求解八皇后问题

# 棋盘尺寸
BOARD_SIZE = 8

solution_count = 0
queen_list = [0] * BOARD_SIZE


def eight_queens(cur_column: int):
    """
    输出所有符合要求的八皇后序列
    用一个长度为8的数组代表棋盘的列，数组的数字则为当前列上皇后所在的行数
    :return:
    """
    if cur_column >= BOARD_SIZE:
        global solution_count
        solution_count += 1
        # 解
        print(queen_list)
    else:
        for i in range(BOARD_SIZE):
            if is_valid_pos(cur_column, i):
                queen_list[cur_column] = i
                eight_queens(cur_column + 1)


def is_valid_pos(cur_column: int, pos: int) -> bool:
    """
    因为采取的是每列放置1个皇后的做法
    所以检查的时候不必检查列的合法性，只需要检查行和对角
    1. 行：检查数组在下标为cur_column之前的元素是否已存在pos
    2. 对角：检查数组在下标为cur_column之前的元素，其行的间距pos - QUEEN_LIST[i]
       和列的间距cur_column - i是否一致
    :param cur_column:
    :param pos:
    :return:
    """
    i = 0
    while i < cur_column:
        # 同行
        if queen_list[i] == pos:
            return False
        # 对角线
        if cur_column - i == abs(pos - queen_list[i]):
            return False
        i += 1
    return True


if __name__ == '__main__':
    print('--- eight queens sequence ---')
    eight_queens(0)

    print('\n--- solution count ---')
    print(solution_count)

# 利用回溯算法求解 0-1 背包问题

# 回溯
from typing import List

# 背包选取的物品列表
picks = []
picks_with_max_value = []


def bag(capacity: int, cur_weight: int, items_info: List, pick_idx: int):
    """
    回溯法解01背包，穷举
    :param capacity: 背包容量
    :param cur_weight: 背包当前重量
    :param items_info: 物品的重量和价值信息
    :param pick_idx: 当前物品的索引
    :return:
    """
    # 考察完所有物品，或者在中途已经装满
    if pick_idx >= len(items_info) or cur_weight == capacity:
        global picks_with_max_value
        if get_value(items_info, picks) > \
                get_value(items_info, picks_with_max_value):
            picks_with_max_value = picks.copy()
    else:
        item_weight = items_info[pick_idx][0]
        if cur_weight + item_weight <= capacity:  # 选
            picks[pick_idx] = 1
            bag(capacity, cur_weight + item_weight, items_info, pick_idx + 1)

        picks[pick_idx] = 0  # 不选
        bag(capacity, cur_weight, items_info, pick_idx + 1)


def get_value(items_info: List, pick_items: List):
    values = [_[1] for _ in items_info]
    return sum([a * b for a, b in zip(values, pick_items)])


if __name__ == '__main__':
    # [(weight, value), ...]
    items_info = [(3, 5), (2, 2), (1, 4), (1, 2), (4, 10)]
    capacity = 8

    print('--- items info ---')
    print(items_info)

    print('\n--- capacity ---')
    print(capacity)

    picks = [0] * len(items_info)
    bag(capacity, 0, items_info, 0)

    print('\n--- picks ---')
    print(picks_with_max_value)

    print('\n--- value ---')
    print(get_value(items_info, picks_with_max_value))

# 利用分治算法求一组数据的逆序对个数

inversion_num = 0

def merge_sort_counting(nums, start, end):
    if start >= end:
        return

    mid = (start + end) // 2
    merge_sort_counting(nums, start, mid)
    merge_sort_counting(nums, mid + 1, end)
    merge(nums, start, mid, end)


def merge(nums, start, mid, end):
    global inversion_num
    i = start
    j = mid + 1
    tmp = []
    while i <= mid and j <= end:
        if nums[i] <= nums[j]:
            inversion_num += j - mid - 1
            tmp.append(nums[i])
            i += 1
        else:
            tmp.append(nums[j])
            j += 1

    while i <= mid:
        # 这时nums[i]的逆序数是整个nums[mid+1: end+1]的长度
        inversion_num += end - mid
        tmp.append(nums[i])
        i += 1

    while j <= end:
        tmp.append(nums[j])
        j += 1

    nums[start: end + 1] = tmp

if __name__ == '__main__':
    print('--- count inversion number using merge sort ---')
    # nums = [5, 0, 4, 2, 3, 1, 6, 8, 7]
    nums = [5, 0, 4, 2, 3, 1, 3, 3, 3, 6, 8, 7]
    print('nums  : {}'.format(nums))
    merge_sort_counting(nums, 0, len(nums) - 1)
    print('sorted: {}'.format(nums))
    print('inversion number: {}'.format(inversion_num))

# 0-1 背包问题

# 动态规划
from typing import List, Tuple

def bag(items_info: List[int], capacity: int):
    """
        固定容量的背包，计算能装进背包的物品组合的最大重量
        :param items_info: 每个物品的重量
        :param capacity: 背包容量
        :return: 最大装载重量
        """
    n = len(items_info)
    memo = [[-1] * (capacity + 1) for i in range(n)]
    memo[0][0] = 1
    if items_info[0] <= capacity:
        memo[0][items_info[0]] = 1

    for i in range(1, n):
        for cur_weight in range(capacity + 1):
            if memo[i - 1][cur_weight] != -1:
                memo[i][cur_weight] = memo[i - 1][cur_weight]  # 不选
                if cur_weight + items_info[i] <= capacity:  # 选
                    memo[i][cur_weight + items_info[i]] = 1

    for w in range(capacity, -1, -1):
        if memo[-1][w] != -1:
            return w


def bag_with_max_value(items_info: List[Tuple[int, int]], capacity: int) -> int:
    """
    固定容量的背包，计算能装进背包的物品组合的最大价值
    :param items_info: 物品的重量和价值
    :param capacity: 背包容量
    :return: 最大装载价值
    """
    n = len(items_info)
    memo = [[-1] * (capacity + 1) for i in range(n)]
    memo[0][0] = 0
    if items_info[0][0] <= capacity:
        memo[0][items_info[0][0]] = items_info[0][1]

    for i in range(1, n):
        for cur_weight in range(capacity + 1):
            if memo[i - 1][cur_weight] != -1:
                memo[i][cur_weight] = memo[i - 1][cur_weight]
                if cur_weight + items_info[i][0] <= capacity:
                    memo[i][cur_weight + items_info[i][0]] = max(memo[i][cur_weight + items_info[i][0]],
                                                                 memo[i - 1][cur_weight] + items_info[i][1])
    return max(memo[-1])


if __name__ == '__main__':
    # [weight, ...]
    items_info = [2, 2, 4, 6, 3]
    capacity = 9
    print(bag(items_info, capacity))

    # [(weight, value), ...]
    items_info = [(3, 5), (2, 2), (1, 4), (1, 2), (4, 10)]
    capacity = 8
    print(bag_with_max_value(items_info, capacity))

# 最小路径和（详细可看 Minimum Path Sum）

from typing import List
from itertools import accumulate

def min_dist(weights: List[List[int]]) -> int:
    """Find the minimum weight path from the weights matrix."""
    m, n = len(weights), len(weights[0])
    table = [[0] * n for _ in range(m)]
    # table[i][j] is the minimum distance (weight) when
    # there are i vertical moves and j horizontal moves
    # left.
    table[0] = list(accumulate(reversed(weights[-1])))
    for i, v in enumerate(accumulate(row[-1] for row in reversed(weights))):
        table[i][0] = v
    for i in range(1, m):
        for j in range(1, n):
            table[i][j] = weights[~i][~j] + min(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def min_dist_recur(weights: List[List[int]]) -> int:
    m, n = len(weights), len(weights[0])
    table = [[0] * n for _ in range(m)]

    def min_dist_to(i: int, j: int) -> int:
        if i == j == 0: return weights[0][0]
        if table[i][j]: return table[i][j]
        min_left = float("inf") if j - 1 < 0 else min_dist_to(i, j - 1)
        min_up = float("inf") if i - 1 < 0 else min_dist_to(i - 1, j)
        return weights[i][j] + min(min_left, min_up)

    return min_dist_to(m - 1, n - 1)


if __name__ == "__main__":
    weights = [[1, 3, 5, 9], [2, 1, 3, 4], [5, 2, 6, 7], [6, 8, 4, 3]]
    print(min_dist(weights))
    print(min_dist_recur(weights))

# 编程实现莱文斯坦最短编辑距离

def levenshtein_dp(s: str, t: str) -> int:
    m, n = len(s), len(t)
    table = [[0] * (n) for _ in range(m)]

    for i in range(n):
        if s[0] == t[i]:
            table[0][i] = i - 0
        elif i != 0:
            table[0][i] = table[0][i - 1] + 1
        else:
            table[0][i] = 1

    for i in range(m):
        if s[i] == t[0]:
            table[i][0] = i - 0
        elif i != 0:
            table[i][0] = table[i - 1][0] + 1
        else:
            table[i][0] = 1

    for i in range(1, m):
        for j in range(1, n):
            table[i][j] = min(1 + table[i - 1][j], 1 + table[i][j - 1], int(s[i] != t[j]) + table[i - 1][j - 1])

    print(table)
    return table[-1][-1]


def common_substring_dp(s: str, t: str) -> int:
    m, n = len(s), len(t)
    table = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            table[i][j] = max(table[i - 1][j], table[i][j - 1], int(s[i - 1] == t[j - 1]) + table[i - 1][j - 1])
    return table[-1][-1]


if __name__ == "__main__":
    s = "mitcmu"
    t = "mtacnu"

    print(levenshtein_dp(s, t))
    print(common_substring_dp(s, t))

    s = "kitten"
    t = "sitting"

    print(levenshtein_dp(s, t))
    print(common_substring_dp(s, t))

    s = "flaw"
    t = "lawn"

    print(levenshtein_dp(s, t))
    print(common_substring_dp(s, t))

# 编程实现查找两个字符串的最长公共子序列

'''
最长公共子序列的状态转移方程为：（和最短编辑距离类似，只和左方，上方，左上方的元素有关）
①当str1[i] == str2[j]时：dp[i][j] = dp[i-1][j-1] + 1
②当str1[i] != str2[j]时：dp[i][j] = max(dp[i][j-1], dp[i-1][j])
'''
def LCS(str1, str2):
    # 寻找str1和str2的最长公共子序列
    len1 = len(str1); len2 = len(str2)
    dp = [[0 for x in range(len2)] for y in range(len1)]   # 开辟一个二维数组dp，初始全为0
    dp[0][0] = 1 if str1[0] == str2[0] else 0
    for col in range(1,len2):
        dp[0][col] = 1 if str1[0] == str2[col] else dp[0][col-1]   # 初始化第一行
    for row in range(1,len1):
        dp[row][0] = 1 if str1[row] == str2[0] else dp[row-1][0]   # 初始化第一列  
    for i in range(1,len1):
        for j in range(1,len2):
            dp[i][j] = max(dp[i][j-1], dp[i-1][j]) if str1[i] != str2[j] else dp[i-1][j-1] + 1
    
    return dp[-1][-1]

# 编程实现一个数据序列的最长递增子序列

from typing import List


def longest_increasing_subsequence(nums: List[int]) -> int:
    """
    最长子上升序列的一种DP解法，从回溯解法转化，思路类似于有限物品的背包问题
    每一次决策都算出当前可能的lis的长度，重复子问题合并，合并策略是lis的末尾元素最小
    时间复杂度：O(n^2)
    空间复杂度：O(n^2)，可优化至O(n)
    没leetcode上的参考答案高效，提供另一种思路作为参考
    https://leetcode.com/problems/longest-increasing-subsequence/solution/
    :param nums:
    :return:
    """
    if not nums:
        return 0

    n = len(nums)
    # memo[i][j] 表示第i次决策，长度为j的lis的 最小的 末尾元素数值
    # 每次决策都根据上次决策的所有可能转化，空间上可以类似背包优化为O(n)
    memo = [[-1] * (n + 1) for _ in range(n)]

    # 第一列全赋值为0，表示每次决策都不选任何数
    for i in range(n):
        memo[i][0] = 0
    # 第一次决策选数组中的第一个数
    memo[0][1] = nums[0]

    for i in range(1, n):
        for j in range(1, n + 1):
            # case 1: 长度为j的lis在上次决策后存在，nums[i]比长度为j-1的lis末尾元素大
            if memo[i - 1][j] != -1 and nums[i] > memo[i - 1][j - 1]:
                memo[i][j] = min(nums[i], memo[i - 1][j])

            # case 2: 长度为j的lis在上次决策后存在，nums[i]比长度为j-1的lis末尾元素小/等
            if memo[i - 1][j] != -1 and nums[i] <= memo[i - 1][j - 1]:
                memo[i][j] = memo[i - 1][j]

            if memo[i - 1][j] == -1:
                # case 3: 长度为j的lis不存在，nums[i]比长度为j-1的lis末尾元素大
                if nums[i] > memo[i - 1][j - 1]:
                    memo[i][j] = nums[i]
                # case 4: 长度为j的lis不存在，nums[i]比长度为j-1的lis末尾元素小/等
                break

    for i in range(n, -1, -1):
        if memo[-1][i] != -1:
            return i

if __name__ == '__main__':
    # 要求输入的都是大于0的正整数(可优化至支持任意整数)
    nums = [2, 9, 3, 6, 5, 1, 7]
    print(longest_increasing_subsequence(nums))