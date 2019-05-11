# 实现一个支持动态扩容的数组

class DynamicArray:
    
    def __init__ (self):
        'Create an empty array.'
        self._n = 0 #size
        self._capacity = 10    #先给个10
        self._A = self._make_array(self._capacity)
        
    def __len__ (self):
        return self._n
    
    def is_empty(self):
        return self._n == 0
    
    # O(1)
    def __getitem__ (self, k):
        if not 0 <= k < self._n:
            raise ValueError('invalid index') 
        return self._A[k]
       
    # O(1) 
    def append(self, obj):
        if self._n == self._capacity:    #首先要判断该容器是否放得下
            self._resize(2 * self._capacity)
        self._A[self._n] = obj    
        self._n += 1
        
    def _make_array(self, c):
        return (c * ctypes.py_object)( )
    
    def _resize(self, c):
        B = self._make_array(c)
        for k in range(self._n):
            B[k] = self._A[k]
        self._A = B
        self._capacity = c   

    # O(n)
    def insert(self, k, value):
        if self._n == self._capacity:
            self._resize(2 * self._capacity)
        for j in range(self._n, k, -1):    #从后往前一个一个往后移
            self._A[j] = self._A[j-1]
        self._A[k] = value
        self._n += 1
     
    # O(n)    
    def remove(self, value):
        for k in range(self._n):
            if self._A[k] == value:     #一个个查value
                for j in range(k, self._n - 1):
                    self._A[j] = self._A[j+1]   ##再一个个移上来
                self._A[self._n - 1] = None
                self._n -= 1
                return
        raise ValueError( 'value not found' )
    
    def _print(self):
        for i in range(self._n):
            print(self._A[i], end = ' ')
        print()


mylist = DynamicArray()
print('size was: ', str(len(mylist)))
mylist.append(10)
mylist.append(20)
mylist.append(30)
mylist.insert(0, 0)
mylist.insert(1, 5)
mylist.insert(3, 15)
mylist._print()
mylist.remove(20)
mylist._print()
print('size is: ', str(len(mylist)))

# 实现一个大小固定的有序数组，支持动态增删改操作

class MyArray {
  constructor(capacity = 10) {
    this.data = new Array(capacity);
    this.size = 0;
  }

  // 查
  find(index) {
    if (index < 0 || index >= this.size) {
      throw new Error('find error. index < 0 || index >= this.size');
    }

    return this.data[index];
  }

  // 插入
  insert(index, element) {
    if(this.size == this.data.length) {
      this.resize(this.size * 2);
    }

    if (index < 0 || index > this.size) {
      throw new Error('insert error!');
    }

    // 从索引位后往后移一位
    for (let i = index; i < this.size; i++) {
      this.data[i + 1] = this.data[i];
    }

    this.data[index] = element;

    this.size++;
  }

  // 添加
  add(element) {
    this.insert(this.size, element);
  }

  // 删除
  remove(index) {
    if (index < 0 || index >= this.size) {
      throw new Error('remove error');
    }

    let element = this.data[index];

    for (let i = index; i < array.length; i++) {
      this.data[i] = this.data[i + 1];
    }

    this.data[this.size - 1] = undefined;
    this.size--;

    if (Math.floor(this.getCapacity() / 2) === this.size
      && this.size !== 0
    ) {
      this.resize(Math.floor(this.getCapacity() / 2));
    }
    
    return element;
  }

  // 动态扩容
  resize(capacity) {
    const newArray = new Array(capacity);

    for (let i = 0; i < this.size; i++) {
      newArray[i] = this.data[i];
    }

    this.data = newArray;
  }
}

# 实现两个有序数组合并为一个有序数组

const merge = (array1, m, array2, n) => {
  // 交换数组位置和大小
  // 始终保证 n > m
  if (m > n) {
    const temp = array1;
    const temp_size = m;
    m = n;
    n = temp_size;

    array1 = array2;
    array2 = temp;
  }

  let num = m + n - 1;
  --m;
  --n;

  while (m >= 0 && n >= 0) {
    if (array2[n] > array1[m]) {
      array1[num--] = array2[n--];
    } else {
      array1[num--] = array1[m--];
    }
  }

  // 将剩余元素加入到 array1 中
  while(n >= 0) {
    array1[num--] = array2[n--];
  }

  return array1;
};

# 学习哈希表思想，并完成leetcode上的两数之和(1)及Happy Number(202)！(要求全部用哈希思想实现！)（选做）

