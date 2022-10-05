//# include <sys/time.h>
#include <windows.h>
#include <iostream>
#include<stdio.h>
#include<stdlib.h>
#include <iomanip>
#include <sstream>
#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



using namespace std;


unsigned int Act[8399*264] = { 0 };
unsigned int Pas[8399*264] = { 0 };

const int Num = 263;
const int pasNum = 4535;
const int lieNum = 8399;


//
//unsigned int Act[23045*722] = { 0 };
//unsigned int Pas[23045*722] = { 0 };
//
//const int Num = 721;
//const int pasNum = 14325;
//const int lieNum = 23045;


//
//unsigned int Act[37960*1188] = { 0 };
//unsigned int Pas[37960*1188] = { 0 };
//
//const int Num = 1187;
//const int pasNum = 14921;
//const int lieNum = 37960;
//



//unsigned int Act[43577*1363] = { 0 };
//unsigned int Pas[54274*1363] = { 0 };
//
//const int Num = 1362;
//const int pasNum = 54274;
//const int lieNum = 43577;
//

//消元子初始化
void init_A()
{
    //每个消元子第一个为1位所在的位置，就是它所在二维数组的行号
    //例如：消元子（561，...）由Act[561][]存放
    unsigned int a;
    ifstream infile("act.txt");
    char fin[10000] = { 0 };
    int index;
    //从文件中提取行
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;

        //从行中提取单个的数字
        while (line >> a)
        {
            if (biaoji == 0)
            {
                //取每行第一个数字为行标
                index = a;
                biaoji = 1;
            }
            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Act[index * (Num + 1) + Num - 1 - j] += temp;
            Act[index * (Num + 1) + Num] = 1;//设置该位置记录消元子该行是否为空，为空则是0，否则为1
        }
    }
}

//被消元行初始化
void init_P()
{
    //直接按照磁盘文件的顺序存，在磁盘文件是第几行，在数组就是第几行
    unsigned int a;
    ifstream infile("pas.txt");
    char fin[10000] = { 0 };
    int index = 0;
    //从文件中提取行
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;

        //从行中提取单个的数字
        while (line >> a)
        {
            if (biaoji == 0)
            {
                //用Pas[ ][263]存放被消元行每行第一个数字，用于之后的消元操作
                Pas[index * (Num + 1) + Num] = a;
                biaoji = 1;
            }

            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Pas[index * (Num + 1) + Num - 1 - j] += temp;
        }
        index++;
    }
}



__global__ void work(int g_Num, int g_pasNum, int g_lieNum, int* g_Act, int* g_Pas)
{
    int g_index = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = gridDim.x * blockDim.x;

    for (int i = g_lieNum - 1; i - 8 >= -1; i -= 8)
    {
        //每轮处理8个消元子，范围：首项在 i-7 到 i

        for (int j = g_index; j < g_pasNum; j+=gridStride)
        {
            //看被消元行有没有首项在此范围内的
            while (g_Pas[j * (g_Num + 1) + g_Num] <= i && g_Pas[j * (Num + 1) + g_Num] >= i - 7)
            {
                int index = g_Pas[j * (Num + 1) + g_Num];

                if (g_Act[index * (Num + 1) + g_Num] == 1)//消元子不为空
                {
                    //Pas[j][]和Act[（Pas[j][x]）][]做异或
                    //*******************SIMD优化部分***********************
                    //********
                    for (int k = 0; k < g_Num; k ++)
                    {
                        g_Pas[j * (Num + 1) + k] = g_Pas[j * (Num + 1) + k] ^ g_Act[index * (Num + 1) + k];
                    }
                    //*******
                    //********************SIMD优化部分***********************


                    //更新Pas[j][18]存的首项值
                    //做完异或之后继续找这个数的首项，存到Pas[j][18]，若还在范围里会继续while循环
                    //找异或之后Pas[j][ ]的首项
                    int num = 0, S_num = 0;
                    for (num = 0; num < g_Num; num++)
                    {
                        if (g_Pas[j * (Num + 1) + num] != 0)
                        {
                            unsigned int temp = g_Pas[j * (Num + 1) + num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    g_Pas[j * (Num + 1) + g_Num] = S_num - 1;
                }
                else//消元子为空
                {
                    break;
                }
            }
        }
    }

    for (int i = g_lieNum % 8 - 1; i >= 0; i--)
    {
        //每轮处理1个消元子，范围：首项等于i

        for (int j = g_index; j < g_pasNum; j+=gridStride)
        {
            //看53个被消元行有没有首项等于i的
            while (g_Pas[j * (Num + 1) + g_Num] == i)
            {
                if (g_Act[i * (Num + 1) + g_Num] == 1)//消元子不为空
                {
                    //Pas[j][]和Act[i][]做异或
                    //*******************SIMD优化部分***********************
                    //********
                    for (int k = 0; k < g_Num; k ++)
                    {
                        g_Pas[j * (Num + 1) + k] = g_Pas[j * (Num + 1) + k] ^ g_Act[i * (Num + 1) + k];
                    }
                    //*******
                    //********************SIMD优化部分***********************

                    //更新Pas[j][18]存的首项值
                    //做完异或之后继续找这个数的首项，存到Pas[j][18]，若还在范围里会继续while循环
                    //找异或之后Pas[j][ ]的首项
                    int num = 0, S_num = 0;
                    for (num = 0; num < g_Num; num++)
                    {
                        if (g_Pas[j * (Num + 1) + num] != 0)
                        {
                            unsigned int temp = g_Pas[j * (Num + 1) + num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    g_Pas[j * (Num + 1) + g_Num] = S_num - 1;

                }
                else//消元子为空
                {
                    break;
                }
            }
        }
    }

}


//__global__ void upgrade(int g_Num, int g_pasNum, int g_lieNum, int* g_Act, int* g_Pas)
//{
//    
//    printf("%d\n", g_Pas[2 * (g_Num + 1) + g_Num]);
//    g_Pas[2 * (g_Num + 1) + g_Num] = 100;
//
//}


int main()
{
    cudaError_t ret;

    init_A();
    init_P();
    

   /* for (int i = 0; i < lieNum; i++)
    {
        for (int j = 0; j < Num + 1; j++)
        {
            cout << Pas[i * (Num + 1) + j] << " ";
        }
        cout << endl;
    }*/



    //cout << Pas[2 * (Num + 1) + Num] << " ";


    int* g_Act, *g_Pas;
    
    ret=cudaMalloc(&g_Act, lieNum * (Num + 1) * sizeof(int));
    ret=cudaMalloc(&g_Pas, lieNum * (Num + 1) * sizeof(int));
    if (ret != cudaSuccess) {
        printf("cudaMalloc gpudata failed!\n");
    }


    

    size_t threads_per_block = 256;
    size_t number_of_blocks = 32;


    cudaEvent_t start, stop;//计时器
    float etime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);//开始计时

    bool sign;
    do
    {
        ret = cudaMemcpy(g_Act, Act, sizeof(int) * lieNum * (Num + 1), cudaMemcpyHostToDevice);
        ret = cudaMemcpy(g_Pas, Pas, sizeof(int) * lieNum * (Num + 1), cudaMemcpyHostToDevice);
        if (ret != cudaSuccess) {
            printf("cudaMemcpyHostToDevice failed!\n");
        }
        

        //不升格地处理被消元行------------
        work <<< number_of_blocks, threads_per_block >>> (Num, pasNum, lieNum, g_Act, g_Pas);
        //work << < 1024, 100 >> > (Num, pasNum, lieNum, g_Act, g_Pas);

        cudaDeviceSynchronize();
        //不升格地处理被消元行------------


        ret = cudaMemcpy(Act, g_Act, sizeof(int) * lieNum * (Num + 1), cudaMemcpyDeviceToHost);
        ret = cudaMemcpy(Pas, g_Pas, sizeof(int) * lieNum * (Num + 1), cudaMemcpyDeviceToHost);
        if (ret != cudaSuccess) {
            printf("cudaMemcpyDeviceToHost failed!\n");
        }

        //升格消元子，然后判断是否结束
        sign = false;
        for (int i = 0; i < pasNum; i++)
        {
            //找到第i个被消元行的首项
            int temp = Pas[i* (Num + 1) + Num];
            if (temp == -1)
            {
                //说明他已经被升格为消元子了
                continue;
            }
            //看这个首项对应的消元子是不是为空，若为空，则补齐
            if (Act[temp * (Num + 1) + Num] == 0)
            {
                //补齐消元子
                for (int k = 0; k < Num; k++)
                    Act[temp * (Num + 1) + k] = Pas[i * (Num + 1) + k];
                //将被消元行升格
                Pas[i * (Num + 1) + Num] = -1;
                //标志bool设为true，说明此轮还需继续
                sign = true;
            }
        }
    } while (sign == true);
    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);//停止计时
    cudaEventElapsedTime(&etime, start, stop);
    printf("GPU_LU:%f ms\n", etime);




}




