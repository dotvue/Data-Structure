# 实现一个基于链表法解决冲突问题的散列表

class Dict:
    def __init__(self, num):
        self.__solts__ = []
        self.num = num
        for _ in range(num):
            self.__solts__.append([])
    def hash_fun(self,key,num):
        hashval = 0
        x = key
        if x < 0:
            print ("the key is low")
            return
        while x != 0:
            hashval = (hashval << 3) + x%10
            x /=10
        return hashval % num
    def put(self, key, value):
        i = self.hash_fun(key,self.num) % self.num
        for p, (k, v) in enumerate(self.__solts__[i]):
            if k == key:
                break
        else:
            self.__solts__[i].append((key, value))
            return
        self.__solts__[i][p] = (key, value)
    def get(self, key):
        i = self.hash_fun(key,self.num) % self.num
        for k, v in self.__solts__[i]:
            if k == key:
                return v
        raise KeyError(key)
    # keys函数
    def keys(self):
        ret = []
        for solt in self.__solts__:
            for k, _ in solt:
                ret.append(k)
        return ret
    def __getitem__(self,key):
        return self.get(key)

    def __setitem__(self,key,data):
        self.put(key,data)

# 实现一个 LRU 缓存淘汰算法

class LRUcache:
    def __init__(self, size=3):
        self.cache = {}
        self.keys = []
        self.size = size

    def get(self, key):
        if key in self.cache:
            self.keys.remove(key)
            self.keys.insert(0, key)
            return self.cache[key]
        else:
            return None

    def set(self, key, value):
        if key in self.cache:
            self.keys.remove(key)
            self.keys.insert(0, key)
            self.cache[key] = value
        elif len(self.keys) == self.size:
            old = self.keys.pop()
            self.cache.pop(old)
            self.keys.insert(0, key)
            self.cache[key] = value
        else:
            self.keys.insert(0, key)
            self.cache[key] = value

if __name__ == '__main__':
    test = LRUcache()
    test.set('a',2)
    test.set('b',2)
    test.set('c',2)
    test.set('d',2)
    test.set('e',2)
    test.set('f',2)
    print(test.get('c')) # None
    print(test.get('b')) # None
    print(test.get('a')) # None
    print(test.get('e')) # 2