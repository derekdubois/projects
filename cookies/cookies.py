class Jar:
    def __init__(self, capacity=12):
        if capacity < 0:
            raise ValueError('Invalid Capacity')
        self._capacity = capacity
        self._size = 0

    def __str__(self):
        return self.size * '🍪'

    def deposit(self, n):
        if n > (self.capacity - self.size):
            raise ValueError('Too Many Cookies')
        self._size += n

    def withdraw(self, n):
        if n > self.size:
            raise ValueError('Why YOu So HUUUUNgry')
        self._size -= n

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return self._size

jar = Jar()
jar.deposit(6)
jar.withdraw(3)
print(jar)