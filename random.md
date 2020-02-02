# import random
### 1.random.sample(range(num1), num2)
从range(num1)中随机挑选num2个数，顺序随机

    list = [0,1,2,3,4]
    rs = random.sample(list, 2)
输出：[2, 4]    #此数组随着不同的执行，里面的元素随机，但都是两个

    random.sample(range(5), 5)    
输出：[2, 4, 0, 1, 3]
