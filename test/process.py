from multiprocessing import Process, Value, Array

def writer(val, arr):
    val.value = 3.14
    for i in range(len(arr)):
        arr[i] = i * 2

def reader(val, arr):
    print(val.value)
    print(arr[:])

if __name__ == "__main__":
    num = Value('d', 0.0)  # 'd' 表示 double 类型
    arr = Array('i', range(10))  # 'i' 表示整数类型
    p1 = Process(target=writer, args=(num, arr))
    p2 = Process(target=reader, args=(num, arr))
    p1.start()
    p2.start()
    p1.join()
    p2.join()