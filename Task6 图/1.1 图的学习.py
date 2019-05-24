# 实现有向图、无向图、有权图、无权图的邻接矩阵和邻接表表示方法

class Graph_Matrix:
    """
    Adjacency Matrix
    """
    def __init__(self, vertices=[], matrix=[]):
        """
        :param vertices:a dict with vertex id and index of matrix , such as {vertex:index}
        :param matrix: a matrix
        """
        self.matrix = matrix
        self.edges_dict = {}  # {(tail, head):weight}
        self.edges_array = []  # (tail, head, weight)
        self.vertices = vertices
        self.num_edges = 0
 
        # if provide adjacency matrix then create the edges list
        if len(matrix) > 0:
            if len(vertices) != len(matrix):
                raise IndexError
            self.edges = self.getAllEdges()
            self.num_edges = len(self.edges)
 
        # if do not provide a adjacency matrix, but provide the vertices list, build a matrix with 0
        elif len(vertices) > 0:
            self.matrix = [[0 for col in range(len(vertices))] for row in range(len(vertices))]
 
        self.num_vertices = len(self.matrix)
 
    def isOutRange(self, x):
        try:
            if x >= self.num_vertices or x <= 0:
                raise IndexError
        except IndexError:
            print("节点下标出界")
 
    def isEmpty(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices == 0
 
    def add_vertex(self, key):
        if key not in self.vertices:
            self.vertices[key] = len(self.vertices) + 1
 
        # add a vertex mean add a row and a column
        # add a column for every row
        for i in range(self.getVerticesNumbers()):
            self.matrix[i].append(0)
 
        self.num_vertices += 1
 
        nRow = [0] * self.num_vertices
        self.matrix.append(nRow)
 
    def getVertex(self, key):
        pass
 
    def add_edges_from_list(self, edges_list):  # edges_list : [(tail, head, weight),()]
        for i in range(len(edges_list)):
            self.add_edge(edges_list[i][0], edges_list[i][1], edges_list[i][2], )
 
    def add_edge(self, tail, head, cost=0):
        # if self.vertices.index(tail) >= 0:
        #   self.addVertex(tail)
        if tail not in self.vertices:
            self.add_vertex(tail)
        # if self.vertices.index(head) >= 0:
        #   self.addVertex(head)
        if head not in self.vertices:
            self.add_vertex(head)
 
        # for directory matrix
        self.matrix[self.vertices.index(tail)][self.vertices.index(head)] = cost
        # for non-directory matrix
        # self.matrix[self.vertices.index(fromV)][self.vertices.index(toV)] = \
        #   self.matrix[self.vertices.index(toV)][self.vertices.index(fromV)] = cost
 
        self.edges_dict[(tail, head)] = cost
        self.edges_array.append((tail, head, cost))
        self.num_edges = len(self.edges_dict)
 
    def getEdges(self, V):
        pass
 
    def getVerticesNumbers(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices
 
    def getAllVertices(self):
        return self.vertices
 
    def getAllEdges(self):
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if 0 < self.matrix[i][j] < float('inf'):
                    self.edges_dict[self.vertices[i], self.vertices[j]] = self.matrix[i][j]
                    self.edges_array.append([self.vertices[i], self.vertices[j], self.matrix[i][j]])
 
        return self.edges_array
 
    def __repr__(self):
        return str(''.join(str(i) for i in self.matrix))
 
    def to_do_vertex(self, i):
        print('vertex: %s' % (self.vertices[i]))
 
    def to_do_edge(self, w, k):
        print('edge tail: %s, edge head: %s, weight: %s' % (self.vertices[w], self.vertices[k], str(self.matrix[w][k])))

    # 无向图的邻接矩阵表示方法
    def create_undirected_matrix(my_graph):
        nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    
        matrix = [[0, 1, 1, 1, 1, 1, 0, 0],  # a
                [0, 0, 1, 0, 1, 0, 0, 0],  # b
                [0, 0, 0, 1, 0, 0, 0, 0],  # c
                [0, 0, 0, 0, 1, 0, 0, 0],  # d
                [0, 0, 0, 0, 0, 1, 0, 0],  # e
                [0, 0, 1, 0, 0, 0, 1, 1],  # f
                [0, 0, 0, 0, 0, 1, 0, 1],  # g
                [0, 0, 0, 0, 0, 1, 1, 0]]  # h
    
        my_graph = Graph_Matrix(nodes, matrix)
        print(my_graph)
        return my_graph

    # 显示无向图
    def draw_undircted_graph(my_graph):
    G = nx.Graph()  # 建立一个空的无向图G
    for node in my_graph.vertices:
        G.add_node(str(node))
    for edge in my_graph.edges:
        G.add_edge(str(edge[0]), str(edge[1]))
 
    print("nodes:", G.nodes())  # 输出全部的节点： [1, 2, 3]
    print("edges:", G.edges())  # 输出全部的边：[(2, 3)]
    print("number of edges:", G.number_of_edges())  # 输出边的数量：1
    nx.draw(G, with_labels=True)
    plt.savefig("undirected_graph.png")
    plt.show()

    # 有向图的邻接矩阵表示方法
    def create_directed_matrix(my_graph):
    nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    inf = float('inf')
    matrix = [[0, 2, 1, 3, 9, 4, inf, inf],  # a
              [inf, 0, 4, inf, 3, inf, inf, inf],  # b
              [inf, inf, 0, 8, inf, inf, inf, inf],  # c
              [inf, inf, inf, 0, 7, inf, inf, inf],  # d
              [inf, inf, inf, inf, 0, 5, inf, inf],  # e
              [inf, inf, 2, inf, inf, 0, 2, 2],  # f
              [inf, inf, inf, inf, inf, 1, 0, 6],  # g
              [inf, inf, inf, inf, inf, 9, 8, 0]]  # h
 
    my_graph = Graph_Matrix(nodes, matrix)
    print(my_graph)
    return my_graph

    # 显示有向图
    def draw_directed_graph(my_graph):
    G = nx.DiGraph()  # 建立一个空的无向图G
    for node in my_graph.vertices:
        G.add_node(str(node))
    G.add_weighted_edges_from(my_graph.edges_array)
 
    print("nodes:", G.nodes())  # 输出全部的节点
    print("edges:", G.edges())  # 输出全部的边
    print("number of edges:", G.number_of_edges())  # 输出边的数量
    nx.draw(G, with_labels=True)
    plt.savefig("directed_graph.png")
    plt.show()

# 实现图的深度优先搜索、广度优先搜索

class Graph(object):
 
    def __init__(self,*args,**kwargs):
        self.node_neighbors = {}
        self.visited = {}
 
    def add_nodes(self,nodelist):
 
        for node in nodelist:
            self.add_node(node)
 
    def add_node(self,node):
        if not node in self.nodes():
            self.node_neighbors[node] = []
 
    def add_edge(self,edge):
        u,v = edge
        if(v not in self.node_neighbors[u]) and ( u not in self.node_neighbors[v]):
            self.node_neighbors[u].append(v)
 
            if(u!=v):
                self.node_neighbors[v].append(u)
 
    def nodes(self):
        return self.node_neighbors.keys()
 
    def depth_first_search(self,root=None):
        order = []
        def dfs(node):
            self.visited[node] = True
            order.append(node)
            for n in self.node_neighbors[node]:
                if not n in self.visited:
                    dfs(n)
 
 
        if root:
            dfs(root)
 
        for node in self.nodes():
            if not node in self.visited:
                dfs(node)
 
        print order
        return order
 
    def breadth_first_search(self,root=None):
        queue = []
        order = []
        def bfs():
            while len(queue)> 0:
                node  = queue.pop(0)
 
                self.visited[node] = True
                for n in self.node_neighbors[node]:
                    if (not n in self.visited) and (not n in queue):
                        queue.append(n)
                        order.append(n)
 
        if root:
            queue.append(root)
            order.append(root)
            bfs()
 
        for node in self.nodes():
            if not node in self.visited:
                queue.append(node)
                order.append(node)
                bfs()
        print order
 
        return order
 
 
if __name__ == '__main__':
    g = Graph()
    g.add_nodes([i+1 for i in range(8)])
    g.add_edge((1, 2))
    g.add_edge((1, 3))
    g.add_edge((2, 4))
    g.add_edge((2, 5))
    g.add_edge((4, 8))
    g.add_edge((5, 8))
    g.add_edge((3, 6))
    g.add_edge((3, 7))
    g.add_edge((6, 7))
    print "nodes:", g.nodes()
    
    order = g.breadth_first_search(1)
    order = g.depth_first_search(1)

# 实现 Dijkstra 算法、A* 算法

# Dijkstra 算法 
## 表示无穷大  
INF_val = 9999  
  
class Dijkstra_Path():  
    def __init__(self, node_map):  
        self.node_map = node_map  
        self.node_length = len(node_map)  
        self.used_node_list = []  
        self.collected_node_dict = {}  
      
    def __call__(self, from_node, to_node):  
        self.from_node = from_node  
        self.to_node = to_node  
        self._init_dijkstra()  
        return self._format_path()  
  
    def _init_dijkstra(self):  
        ## Add from_node to used_node_list  
        self.used_node_list.append(self.from_node)  
        for index1 in range(self.node_length):  
            self.collected_node_dict[index1] = [INF_val, -1]  
    
        self.collected_node_dict[self.from_node] = [0, -1] # from_node don't have pre_node  
        for index1, weight_val in enumerate(self.node_map[self.from_node]):  
            if weight_val:  
                self.collected_node_dict[index1] = [weight_val, self.from_node] # [weight_val, pre_node]  
          
        self._foreach_dijkstra()  
      
    def _foreach_dijkstra(self):  
        while(len(self.used_node_list) < self.node_length - 1):  
            min_key = -1  
            min_val = INF_val  
            for key, val in self.collected_node_dict.items(): # 遍历已有权值节点  
                if val[0] < min_val and key not in self.used_node_list:  
                    min_key = key  
                    min_val = val[0]  
  
            ## 把最小的值加入到used_node_list          
            if min_key != -1:  
                self.used_node_list.append(min_key)  
  
            for index1, weight_val in enumerate(self.node_map[min_key]):  
                ## 对刚加入到used_node_list中的节点的相邻点进行遍历比较  
                if weight_val > 0 and self.collected_node_dict[index1][0] > weight_val + min_val:  
                    self.collected_node_dict[index1][0] = weight_val + min_val # update weight_val  
                    self.collected_node_dict[index1][1] = min_key  
  
  
    def _format_path(self):  
        node_list = []  
        temp_node = self.to_node  
        node_list.append((temp_node, self.collected_node_dict[temp_node][0]))  
        while self.collected_node_dict[temp_node][1] != -1:  
          temp_node = self.collected_node_dict[temp_node][1]  
          node_list.append((temp_node, self.collected_node_dict[temp_node][0]))  
        node_list.reverse()  
        return node_list  
  
def set_node_map(node_map, node, node_list):  
    for x, y, val in node_list:  
        node_map[node.index(x)][node.index(y)] = node_map[node.index(y)][node.index(x)] = val  
  
      
if __name__ == "__main__":  
    node = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  
    node_list = [('A', 'F', 9), ('A', 'B', 10), ('A', 'G', 15), ('B', 'F', 2),  
                 ('G', 'F', 3), ('G', 'E', 12), ('G', 'C', 10), ('C', 'E', 1),  
                 ('E', 'D', 7)]  
    
    ## init node_map to 0  
    node_map = [[0 for val in range(len(node))] for val in range(len(node))]  
    
    ## set node_map  
    set_node_map(node_map, node, node_list)  
    
    ## select one node to obj node, e.g. A --> D(node[0] --> node[3])  
    from_node = node.index('A')  
    to_node = node.index('D')  
    dijkstrapath = Dijkstra_Path(node_map)  
    path = dijkstrapath(from_node, to_node)  
    print(path)

# A* 算法
class Node:
    def __int__(self):
        self.unable = False
        self.distanceFromDes = -1  # 距离终点的距离
        self.distanceFromOri = -1  # 距离起点的距离
        self.allDistance = -1
        self.added = False
        self.closed = False
        self.parent = None
        self.x = -1
        self.y = -1
 
 
def GenerateMap(m, n):
    map = list()
    for j in range(m):
        nodeRow = list()
        map.append(nodeRow)
        for i in range(n):
            node = Node()
            node.y = j
            node.x = i
            node.unable = False
            node.distanceFromDes = -1  # 距离终点的距离
            node.distanceFromOri = -1  # 距离起点的距离
            node.allDistance = -1
            node.added = False
            node.closed = False
            node.parent = None
            nodeRow.append(node)
    return map
 
 
def SetUnableMapNode(map, ls=()):  # 要求一个坐标队列，里边的点上的Node的unable == True
    for index in ls:
        map[index[0]][index[1]].unable = True
    return map
 
 
def GetDistanceFromDes(map, mapSize, desIndex):  # map二维数组，mapsize(m,n),desIndex终点坐标
    for ls in map:
        for node in ls:
            node.added = False
    desNode = map[desIndex[0]][desIndex[1]]
    desNode.distanceFromDes = 0
    addedList = list()  # 已经加入的队列，已有值distanceFromDes
    needList = list()  # 待加入的队列，需要评估值distanceFromDes
    addedList.append(desNode)
    desNode.added = True
    while(len(addedList) != 0):  # 当地图上所有可以遍历的点还没全确定
        while(len(addedList) != 0):  # 当一个大循环中，addedList还没被needList取代
            # 从addedList中选出来的一个点，找needList中的needNode
            mainNode = addedList.pop(0)
            mainDistanceFromDes = mainNode.distanceFromDes
            y = mainNode.y
            x = mainNode.x
            for needNodey in (y + 1, y, y - 1):
                if needNodey < 0 or needNodey >= mapSize[0]:
                    continue
                for needNodex in (x + 1, x, x - 1):
                    if needNodex < 0 or needNodex >= mapSize[1]:
                        continue
                    needNode = map[needNodey][needNodex]  # 坐标不出界
                    if needNode.unable == True or needNode.added == True:
                        continue  # 坐标也满足add的要求
                    yOffset = needNodey - y
                    xOffset = needNodex - x
                    allOffset = yOffset + xOffset
                    if allOffset == 1 or allOffset == -1:
                        distanceFromDes = mainDistanceFromDes + 1
                    elif allOffset == -2 or allOffset == 0 or allOffset == 2:
                        distanceFromDes = mainDistanceFromDes + 1.4
                    else:
                        print("error in needNode's distanceFromDes")
                    
                    if needNode in needList:  # 设置needNode的距离，要求最小
                        if distanceFromDes < needNode.distanceFromDes:
                            needNode.distanceFromDes = distanceFromDes
                    else:  # needNode 不在needList中 distanceFromDes一定是-1
                        needNode.distanceFromDes = distanceFromDes
                        needList.append(needNode)
                    #print(needNode.y,needNode.x,needNode.distanceFromDes)
        # needList 已满 addedList已空
        addedList = needList
        for node in addedList:
            node.added = True
        needList = list()
    return map
 
 
def GetMinDistanceNodeList(map, mapSize, oriIndex, desIndex):
    for ls in map:
        for node in ls:
            node.added = False
    openedList = list()
    node = map[oriIndex[0]][oriIndex[1]]
    node.distanceFromOri = 0
    node.allDistance = 0
    openedList.append(node)
    node.added = True
    while len(openedList) != 0:
        #print('new turn')
        node = openedList.pop(0)
        node.closed = True
        if node.y == desIndex[0] and node.x == desIndex[1]:
            finalListNeedReverse = list()
            while node != None:
                finalListNeedReverse.append(node)
                node = node.parent
            finalListNeedReverse.reverse()
            return finalListNeedReverse
        neighboursList = list()
        y = node.y
        x = node.x
        parentDistanceFromOri = node.distanceFromOri
        for needNodey in (y + 1, y, y - 1):
            if needNodey < 0 or needNodey >= mapSize[0]:
                continue
            for needNodex in (x + 1, x, x - 1):
                if needNodex < 0 or needNodex >= mapSize[1]:
                    continue
                needNode = map[needNodey][needNodex]  # 坐标不出界
                if needNode.unable == True or needNode.closed == True or needNode.added == True:
                    continue  # 坐标也满足add的要求
                yOffset = needNodey - y
                xOffset = needNodex - x
                allOffset = yOffset + xOffset
                if allOffset == 1 or allOffset == -1:
                    distanceFromOri = parentDistanceFromOri + 1
                elif allOffset == -2 or allOffset == 0 or allOffset == 2:
                    distanceFromOri = parentDistanceFromOri + 1.4
                else:
                    print("error in needNode's distanceFromDes")
                if needNode in neighboursList:  # 设置needNode的距离，要求最小
                    if distanceFromOri < needNode.distanceFromOri:
                        needNode.distanceFromOri = distanceFromOri
                else:  # needNode 不在needList中 distanceFromDes一定是-1
                    needNode.distanceFromOri = distanceFromOri
                    neighboursList.append(needNode)  # 距离计算完成
        for needNode in neighboursList:  # 开始添加至openedList
            needNode.parent = node
            needNode.allDistance = needNode.distanceFromOri + needNode.distanceFromDes
            needNode.added = True
            openedList.append(needNode)
            #print(needNode.x,needNode.y,needNode.allDistance)
        openedList.sort(key=lambda x: x.allDistance)  # 最小距离的排在前边
    print("Cant find any way to the destination!")
    return None
 
 
def main():
    TestGetDistanceFromDes()
 
 
def TestGetDistanceFromDes():
    m = 27 #设置地图的长
    n = 25 #设置地图的宽
    oriIndex = (0, 0) #设置起点坐标
    desIndex = (23, 24) #设置终点坐标
    map = GenerateMap(m, n) #生成地图节点
    obstacleList = [(1,1),(2,1),(3,1),(4,3),(1,3),(2,3),(3,3),(0,1),(5,1),(5,3)] #设置障碍
    map = SetUnableMapNode(map,obstacleList)  #在地图中添加障碍
    GetDistanceFromDes(map, (m, n),desIndex) #添加终点，并计算节点与终点的距离
 
    print("Distance From Destination")
    for nodeRow in map:
        for node in nodeRow:
            if node.distanceFromDes != -1:
                print('{:^5.1f}'.format(node.distanceFromDes),end = " ")
            else:
                print('  X  ',end = " ")
        print()
    print()
 
    TestGetMinDistanceNodeList(map,(m,n),oriIndex,desIndex) #终点距离测试完了，进入下一阶段
 
def TestGetMinDistanceNodeList(map, mapSize, oriIndex,desIndex):
    finalList = GetMinDistanceNodeList(map, mapSize, oriIndex, desIndex) #添加起点，并生成起点到终点的节点队列
 
    directions = (('↘','↓','↙'),('→',"S",'←'),('↗','↑','↖'))
    print('How To Go')
    for nodeRow in map:
        for node in nodeRow:
            if node in finalList:
                #print('  *  ',end ='')
                parent = node.parent
                if parent != None:
                    if node.y!=desIndex[0] or node.x!=desIndex[1]:
                        (y,x) = (parent.y - node.y+1,parent.x -node.x+1)
                        print('  '+directions[y][x]+'  ',end = '')
                    else:
                        print('Final',end = '')
                else:
                    print('Start',end ='')
            else:
                if node.allDistance != -1:
                    print('{:^4.1f}'.format(node.allDistance),end = " ")
                else:
                    print('  X  ',end = " ")
        print()
    print()
main()    

# 实现拓扑排序的 Kahn 算法、DFS 算法

# Kahn 算法
def topologicalSort(l):
    indegree = [0] * len(l)
    queue = []
    topo = []
    cnt = 0
 
    for key, values in l.items():
        for i in values:
            indegree[i] += 1
 
    for i in range(len(indegree)):
        if indegree[i] == 0:
            queue.append(i)
 
    while(queue):
        vertex = queue.pop(0)
        cnt += 1
        topo.append(vertex)
        for x in l[vertex]:
            indegree[x] -= 1
            if indegree[x] == 0:
                queue.append(x)
 
    if cnt != len(l):
        print("Cycle exists")
    else:
        print(topo)
 
# Adjacency List of Graph
l = {0:[1,2], 1:[3], 2:[3], 3:[4,5], 4:[], 5:[]}
topologicalSort(l)

# DFS 算法
# f = lambda:map(int, input().split())
# n, m = f()
# g = [[0]*(n+1) for i in [0]*(n+1)]
# for k in range(m):
#     i, j = f()
#     g[i][j] = 1
# 手动输入数据
g = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
 
for i in range(1, len(g)):
    print(g[i])
# 记录拓扑排序顺序
l = []
# 记录出度为0的点
s = []
# 记录已经访问的点
v = []
for i in range(1, len(g)):
    if g[i].__eq__([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
        s.append(i)
 
 
# 好像大概是完成了的，但是应该是有问题的
def visit(g, n):
    if n not in v:
        v.append(n)
    for m in range(1, len(g)):
        if g[m][n] and m not in v:
            visit(g, m)
    # 核心之处，代理解
    l.insert(0, n)
 
 
# 对s中的点进行dfs访问
for n in s:
    visit(g, n)
 
print('拓扑排序：')
print(l)
print('访问节点顺序：')
print(v)
print('出度为0节点集合：')
print(s)
--------------------- 
参考：https://blog.csdn.net/yyy430/article/details/90438924 