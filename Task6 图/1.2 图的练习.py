# Number of Islands（岛屿的个数）
# 英文版：https://leetcode.com/problems/number-of-islands/description/
# 中文版：https://leetcode-cn.com/problems/number-of-islands/description/

class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        output = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]=='1':
                    output += 1
                    self.dfs(grid,i,j)
        return output
    
    def dfs(self,grid,i,j):
        grid[i][j] = '0'
        if i-1>=0 and grid[i-1][j]=='1':
            self.dfs(grid,i-1,j)
        if i+1<len(grid) and grid[i+1][j]=='1':
            self.dfs(grid,i+1,j)
        if j-1>=0 and grid[i][j-1]=='1':
            self.dfs(grid,i,j-1)
        if j+1<len(grid[0]) and grid[i][j+1]=='1':
            self.dfs(grid,i,j+1)

# Valid Sudoku（有效的数独）
# 英文版：https://leetcode.com/problems/valid-sudoku/
# 中文版：https://leetcode-cn.com/problems/valid-sudoku/

class Solution:
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        Cell = [[] for i in range(9)]                   # 没有必要用dict,我们只某个数字关心有没有出现过
        Col =  [[] for i in range(9)]
        Row =  [[] for i in range(9)]
        
        for i,row in enumerate(board):                  # 将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
            for j,num in enumerate(row):
                if num != '.':
                    k = (i//3)*3 + j//3
                    if num in Row[i] + Col[j] + Cell[k]:    # list的骚操作,将三个list顺序的拼接 
                        return False
                    Row[i].append(num)
                    Col[j].append(num)
                    Cell[k].append(num)
                        
        return True