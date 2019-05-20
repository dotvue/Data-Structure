# 两数之和
# 英文版：https://leetcode.com/problems/two-sum/
# 中文版：https://leetcode-cn.com/problems/two-sum/

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dict={}
        for i in range(len(nums)):
            if target-nums[i] not in dict:
                dict[nums[i]]=i
            else:
                return [dict[target-nums[i]], i]