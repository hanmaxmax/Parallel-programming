# include <sys/time.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <mpi.h>
#include <pmmintrin.h>
#include <omp.h>
using namespace std;


static const int thread_count = 4;

/*
unsigned int Act[8399][264] = { 0 };
unsigned int Pas[8399][264] = { 0 };

const int Num = 263;
const int pasNum = 4535;
const int lieNum = 8399;
*/



//
//unsigned int Act[23045][722] = { 0 };
//unsigned int Pas[23045][722] = { 0 };
//
//const int Num = 721;
//const int pasNum = 14325;
//const int lieNum = 23045;


//
//unsigned int Act[37960][1188] = { 0 };
//unsigned int Pas[37960][1188] = { 0 };
//
//const int Num = 1187;
//const int pasNum = 14921;
//const int lieNum = 37960;



unsigned int Act[43577][1363] = { 0 };
unsigned int Pas[54274][1363] = { 0 };

const int Num = 1362;
const int pasNum = 54274;
const int lieNum = 43577;



//消元子初始化
void init_A()
{
    //每个消元子第一个为1位所在的位置，就是它所在二维数组的行号
    //例如：消元子（561，...）由Act[561][]存放
    unsigned int a;
    ifstream infile("act3.txt");
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
            Act[index][Num - 1 - j] += temp;
            Act[index][Num] = 1;//设置该位置记录消元子该行是否为空，为空则是0，否则为1
        }
    }
}

//被消元行初始化
void init_P()
{
    //直接按照磁盘文件的顺序存，在磁盘文件是第几行，在数组就是第几行
    unsigned int a;
    ifstream infile("pas3.txt");
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
                Pas[index][Num] = a;
                biaoji = 1;
            }

            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Pas[index][Num - 1 - j] += temp;
        }
        index++;
    }
}



void f_ordinary()
{
    timeval t_start;
    timeval t_end;
    gettimeofday(&t_start, NULL);

    bool sign;
    do
    {
        //不升格地处理被消元行------------------------------------------------------
        //---------------------------begin-------------------------------------

        int i;
        for (i = lieNum - 1; i - 8 >= -1; i -= 8)
        {
            //每轮处理8个消元子，范围：首项在 i-7 到 i
            for (int j = 0; j < pasNum; j++)
            {
                //看被消元行有没有首项在此范围内的
                while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
                {
                    int index = Pas[j][Num];

                    if (Act[index][Num] == 1)//消元子不为空
                    {
                        //Pas[j][]和Act[（Pas[j][x]）][]做异或
                        for (int k = 0; k < Num; k ++)
                        {
                            Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        }




                        //更新Pas[j][18]存的首项值
                        //做完异或之后继续找这个数的首项，存到Pas[j][18]，若还在范围里会继续while循环
                        //找异或之后Pas[j][ ]的首项
                        int num = 0, S_num = 0;
                        for (num = 0; num < Num; num++)
                        {
                            if (Pas[j][num] != 0)
                            {
                                unsigned int temp = Pas[j][num];
                                while (temp != 0)
                                {
                                    temp = temp >> 1;
                                    S_num++;
                                }
                                S_num += num * 32;
                                break;
                            }
                        }
                        Pas[j][Num] = S_num - 1;
                    }
                    else//消元子为空
                    {
                        break;
                    }
                }
            }
        }

        for (i = i + 8; i >= 0; i--)
        {
            //每轮处理1个消元子，范围：首项等于i

            for (int j = 0; j < pasNum; j++)
            {
                //看53个被消元行有没有首项等于i的
                while (Pas[j][Num] == i)
                {
                    if (Act[i][Num] == 1)//消元子不为空
                    {
                        //Pas[j][]和Act[i][]做异或
                        for (int k = 0; k < Num; k ++)
                        {
                            Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        }

                        //更新Pas[j][18]存的首项值
                        //做完异或之后继续找这个数的首项，存到Pas[j][18]，若还在范围里会继续while循环
                        //找异或之后Pas[j][ ]的首项
                        int num = 0, S_num = 0;
                        for (num = 0; num < Num; num++)
                        {
                            if (Pas[j][num] != 0)
                            {
                                unsigned int temp = Pas[j][num];
                                while (temp != 0)
                                {
                                    temp = temp >> 1;
                                    S_num++;
                                }
                                S_num += num * 32;
                                break;
                            }
                        }
                        Pas[j][Num] = S_num - 1;

                    }
                    else//消元子为空
                    {
                        break;
                    }
                }
            }
        }

        //----------------------------------end--------------------------------
        //不升格地处理被消元行--------------------------------------------------------



        //升格消元子，然后判断是否结束
        sign = false;
        for (int i = 0; i < pasNum; i++)
        {
            //找到第i个被消元行的首项
            int temp = Pas[i][Num];
            if (temp == -1)
            {
                //说明他已经被升格为消元子了
                continue;
            }

            //看这个首项对应的消元子是不是为空，若为空，则补齐
            if (Act[temp][Num] == 0)
            {
                //补齐消元子
                for (int k = 0; k < Num; k++)
                    Act[temp][k] = Pas[i][k];
                //将被消元行升格
                Pas[i][Num] = -1;
                //标志bool设为true，说明此轮还需继续
                sign = true;
            }
        }

    } while (sign == true);


    gettimeofday(&t_end, NULL);
    cout << "ordinary time cost: "
        << 1000 * (t_end.tv_sec - t_start.tv_sec) +
        0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
}


void f_ordinary1()
{
    timeval t_start;
    timeval t_end;
    gettimeofday(&t_start, NULL);

    int i;
    for (i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        //每轮处理8个消元子，范围：首项在 i-7 到 i

        for (int j = 0; j < pasNum; j++)
        {
            //看4535个被消元行有没有首项在此范围内的
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1)//消元子不为空
                {
                    //Pas[j][]和Act[（Pas[j][18]）][]做异或
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }

                    //更新Pas[j][18]存的首项值
                    //做完异或之后继续找这个数的首项，存到Pas[j][18]，若还在范围里会继续while循环
                    //找异或之后Pas[j][ ]的首项
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else//消元子为空
                {
                    //Pas[j][]来补齐消元子
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];

                    Act[index][Num] = 1;//设置消元子非空
                    break;
                }

            }
        }
    }


    for (int i = lieNum % 8 - 1; i >= 0; i--)
    {
        //每轮处理1个消元子，范围：首项等于i

        for (int j = 0; j < pasNum; j++)
        {
            //看53个被消元行有没有首项等于i的
            while (Pas[j][Num] == i)
            {
                if (Act[i][Num] == 1)//消元子不为空
                {
                    //Pas[j][]和Act[i][]做异或
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }

                    //更新Pas[j][18]存的首项值
                    //做完异或之后继续找这个数的首项，存到Pas[j][18]，若还在范围里会继续while循环
                    //找异或之后Pas[j][ ]的首项
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else//消元子为空
                {
                    //Pas[j][]来补齐消元子
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];

                    Act[i][Num] = 1;//设置消元子非空
                    break;
                }
            }
        }
    }


    gettimeofday(&t_end, NULL);
    cout << "ordinary time cost: "
        << 1000 * (t_end.tv_sec - t_start.tv_sec) +
        0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
}


void super(int rank, int num_proc)
{
    //不升格地处理被消元行------------------------------------------------------
       //---------------------------begin-------------------------------------


    int i;
    
    //每轮处理8个消元子，范围：首项在 i-7 到 i
#pragma omp parallel num_threads(thread_count) 
    for (i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
#pragma omp for schedule(dynamic,20)     
        for (int j = 0; j < pasNum; j++)
        {
            //当前行是自己进程的任务——进行消去
            if (int(j % num_proc) == rank)
            {

                while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
                {
                    int index = Pas[j][Num];

                    if (Act[index][Num] == 1)//消元子不为空
                    {
                        //****************************SIMD******************************
                        ////Pas[j][]和Act[（Pas[j][x]）][]做异或
                        //for (int k = 0; k < Num; k++)
                        //{
                        //    Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        //}
                        int k;
                        __m128 va_Pas, va_Act;
                        for (k = 0; k + 4 <= Num; k += 4)
                        {
                            //Pas[j][k] = Pas[j][k] ^ Act[index][k];
                            va_Pas = _mm_loadu_ps((float*)&(Pas[j][k]));
                            va_Act = _mm_loadu_ps((float*)&(Act[index][k]));

                            va_Pas = _mm_xor_ps(va_Pas, va_Act);
                            _mm_store_ss((float*)&(Pas[j][k]), va_Pas);
                        }

                        for (; k < Num; k++)
                        {
                            Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        }
                        //***************************SIMD******************************


                        //更新Pas[j][18]存的首项值
                        //做完异或之后继续找这个数的首项，存到Pas[j][18]，若还在范围里会继续while循环
                        //找异或之后Pas[j][ ]的首项
                        int num = 0, S_num = 0;
                        for (num = 0; num < Num; num++)
                        {
                            if (Pas[j][num] != 0)
                            {
                                unsigned int temp = Pas[j][num];
                                while (temp != 0)
                                {
                                    temp = temp >> 1;
                                    S_num++;
                                }
                                S_num += num * 32;
                                break;
                            }
                        }
                        Pas[j][Num] = S_num - 1;
                    }
                    else//消元子为空
                    {
                        break;
                    }
                }
            }
        }
    }

#pragma omp parallel num_threads(thread_count) 
    for (int i = lieNum % 8 - 1; i >= 0; i--)
    {
        //每轮处理1个消元子，范围：首项等于i
#pragma omp for schedule(dynamic,20)
        for (int j = 0; j < pasNum; j++)
        {
            //当前行是自己进程的任务——进行消去
            if (int(j % num_proc) == rank)
            {
                while (Pas[j][Num] == i)
                {
                    if (Act[i][Num] == 1)//消元子不为空
                    {
                         //****************************SIMD******************************
                        ////Pas[j][]和Act[（Pas[j][x]）][]做异或
                        //for (int k = 0; k < Num; k++)
                        //{
                        //    Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        //}
                        int k;
                        __m128 va_Pas, va_Act;
                        for (k = 0; k + 4 <= Num; k += 4)
                        {
                            //Pas[j][k] = Pas[j][k] ^ Act[index][k];
                            va_Pas = _mm_loadu_ps((float*)&(Pas[j][k]));
                            va_Act = _mm_loadu_ps((float*)&(Act[i][k]));

                            va_Pas = _mm_xor_ps(va_Pas, va_Act);
                            _mm_store_ss((float*)&(Pas[j][k]), va_Pas);
                        }

                        for (; k < Num; k++)
                        {
                            Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        }
                        //***************************SIMD******************************


                        //更新Pas[j][18]存的首项值
                        //做完异或之后继续找这个数的首项，存到Pas[j][18]，若还在范围里会继续while循环
                        //找异或之后Pas[j][ ]的首项
                        int num = 0, S_num = 0;
                        for (num = 0; num < Num; num++)
                        {
                            if (Pas[j][num] != 0)
                            {
                                unsigned int temp = Pas[j][num];
                                while (temp != 0)
                                {
                                    temp = temp >> 1;
                                    S_num++;
                                }
                                S_num += num * 32;
                                break;
                            }
                        }
                        Pas[j][Num] = S_num - 1;

                    }
                    else//消元子为空
                    {
                        break;
                    }
                }
            }
        }
    }
    //----------------------------------end--------------------------------
    //不升格地处理被消元行--------------------------------------------------------

}


void f_mpi()
{

    int num_proc;//进程数
    int rank;//识别调用进程的rank，值从0~size-1

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //0号进程——任务划分
    if (rank == 0)
    {
        timeval t_start;
        timeval t_end;

        gettimeofday(&t_start, NULL);
        int sign;
        do
        {
            //任务划分
            for (int i = 0; i < pasNum; i++)
            {
                int flag = i % num_proc;
                if (flag == rank)
                    continue;
                else
                    MPI_Send(&Pas[i], Num + 1, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
            }
            super(rank, num_proc);
            //处理完0号进程自己的任务后需接收其他进程处理之后的结果
            for (int i = 0; i < pasNum; i++)
            {
                int flag = i % num_proc;
                if (flag == rank)
                    continue;
                else
                    MPI_Recv(&Pas[i], Num + 1, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            //升格消元子，然后判断是否结束
            sign = 0;
            for (int i = 0; i < pasNum; i++)
            {
                //找到第i个被消元行的首项
                int temp = Pas[i][Num];
                if (temp == -1)
                {
                    //说明他已经被升格为消元子了
                    continue;
                }

                //看这个首项对应的消元子是不是为空，若为空，则补齐
                if (Act[temp][Num] == 0)
                {
                    //补齐消元子
                    for (int k = 0; k < Num; k++)
                        Act[temp][k] = Pas[i][k];
                    //将被消元行升格
                    Pas[i][Num] = -1;
                    //标志设为true，说明此轮还需继续
                    sign = 1;
                }
            }

            for (int r = 1; r < num_proc; r++)
            {
                MPI_Send(&sign, 1, MPI_INT, r, 2, MPI_COMM_WORLD);
            }
            


        } while (sign == 1);

        gettimeofday(&t_end, NULL);
        cout << "super time cost: "
            << 1000 * (t_end.tv_sec - t_start.tv_sec) +
            0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
    }

    //其他进程
    else
    {
        int sign;

        do
        {
            //非0号进程先接收任务
            for (int i = rank; i < pasNum; i += num_proc)
            {
                MPI_Recv(&Pas[i], Num + 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            //执行任务
            super(rank, num_proc);
            //非0号进程完成任务之后，将结果传回到0号进程
            for (int i = rank; i < pasNum; i += num_proc)
            {
                MPI_Send(&Pas[i], Num + 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
            }

            MPI_Recv(&sign, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        } while (sign == 1);
    }
}




int main()
{
    init_A();
    init_P();
    f_ordinary1();

    init_A();
    init_P();
    f_ordinary();


    init_A();
    init_P();
    MPI_Init(NULL, NULL);

    f_mpi();

    MPI_Finalize();

}






