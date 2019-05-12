# Three Sum（求三数之和）
# 英文版：https://leetcode.com/problems/3sum/
# 中文版：https://leetcode-cn.com/problems/3sum/

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums, r = sorted(nums), set()
        for i in [i for i in range(len(nums)-2) if i < 1 or nums[i] > nums[i-1]]:
            d = {-nums[i]-n: j for j, n in enumerate(nums[i + 1:])}
            r.update([(nums[i], n, -nums[i]-n) for j, n in enumerate(nums[i+1:]) if n in d and d[n] > j])
        return list(map(list, r))

# Majority Element（求众数）
# 英文版：https://leetcode.com/problems/majority-element/
# 中文版：https://leetcode-cn.com/problems/majority-element/

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        return sorted(nums)[len(nums) // 2]

# Missing Positive（求缺失的第一个正数）[作为可选]
# 英文版：https://leetcode.com/problems/first-missing-positive/
# 中文版：https://leetcode-cn.com/problems/first-missing-positive/

class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        N = len(nums)
        for i in range(N):
            if nums[i] <= 0:
                nums[i] = N + 1
        ni = 0
        for i in range(N):
            ni = abs(nums[i]) - 1
            if ni < N:
                nums[ni] = -abs(nums[ni])
        for i in range(N):
            if nums[i] > 0:
                return i + 1
        return N + 1