# parallel_homework3
并行程序设计作业3——Pthread编程


## pthread_neon.cpp
除法和消去都划分给多线程，同时让主线程参与多线程运算 并且 控制其他线程的同步

## pthread_neon_1.cpp
普通高斯消去的动态线程实现（neon）

## pthread_neon_2.cpp
普通高斯消去的静态线程+信号量实现，主线程做除法，其余线程消去（neon）

## pthread_neon_3.cpp
普通高斯消去的静态线程+信号量+三重循环全部纳入线程函数实现，主线程只做创建和挂起销毁，其余线程做除法和消去（neon）

## pthread_neon_4.cpp
普通高斯消去的静态线程+barrier实现（neon）

## pthread_sseavx_1.cpp
## pthread_sseavx_2.cpp
## pthread_sseavx_3.cpp
结合x86平台的指令集做pthread

## super.cpp
特殊高斯消去的串行算法2（串行算法1见《SIMD》一节）

## pthread_super.cpp
特殊高斯消去的pthread算法
