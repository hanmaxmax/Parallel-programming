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



//��Ԫ�ӳ�ʼ��
void init_A()
{
    //ÿ����Ԫ�ӵ�һ��Ϊ1λ���ڵ�λ�ã����������ڶ�ά������к�
    //���磺��Ԫ�ӣ�561��...����Act[561][]���
    unsigned int a;
    ifstream infile("act3.txt");
    char fin[10000] = { 0 };
    int index;
    //���ļ�����ȡ��
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;

        //��������ȡ����������
        while (line >> a)
        {
            if (biaoji == 0)
            {
                //ȡÿ�е�һ������Ϊ�б�
                index = a;
                biaoji = 1;
            }
            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Act[index][Num - 1 - j] += temp;
            Act[index][Num] = 1;//���ø�λ�ü�¼��Ԫ�Ӹ����Ƿ�Ϊ�գ�Ϊ������0������Ϊ1
        }
    }
}

//����Ԫ�г�ʼ��
void init_P()
{
    //ֱ�Ӱ��մ����ļ���˳��棬�ڴ����ļ��ǵڼ��У���������ǵڼ���
    unsigned int a;
    ifstream infile("pas3.txt");
    char fin[10000] = { 0 };
    int index = 0;
    //���ļ�����ȡ��
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;

        //��������ȡ����������
        while (line >> a)
        {
            if (biaoji == 0)
            {
                //��Pas[ ][263]��ű���Ԫ��ÿ�е�һ�����֣�����֮�����Ԫ����
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
        //������ش�����Ԫ��------------------------------------------------------
        //---------------------------begin-------------------------------------

        for (int i = lieNum - 1; i - 8 >= -1; i -= 8)
        {
            //ÿ�ִ���8����Ԫ�ӣ���Χ�������� i-7 �� i
            for (int j = 0; j < pasNum; j++)
            {
                //������Ԫ����û�������ڴ˷�Χ�ڵ�
                while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
                {
                    int index = Pas[j][Num];

                    if (Act[index][Num] == 1)//��Ԫ�Ӳ�Ϊ��
                    {
                        //Pas[j][]��Act[��Pas[j][x]��][]�����
                        for (int k = 0; k < Num; k ++)
                        {
                            Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        }




                        //����Pas[j][18]�������ֵ
                        //�������֮������������������浽Pas[j][18]�������ڷ�Χ������whileѭ��
                        //�����֮��Pas[j][ ]������
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
                    else//��Ԫ��Ϊ��
                    {
                        break;
                    }
                }
            }
        }

        for (int i = lieNum % 8 - 1; i >= 0; i--)
        {
            //ÿ�ִ���1����Ԫ�ӣ���Χ���������i

            for (int j = 0; j < pasNum; j++)
            {
                //��53������Ԫ����û���������i��
                while (Pas[j][Num] == i)
                {
                    if (Act[i][Num] == 1)//��Ԫ�Ӳ�Ϊ��
                    {
                        //Pas[j][]��Act[i][]�����
                        for (int k = 0; k < Num; k ++)
                        {
                            Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        }

                        //����Pas[j][18]�������ֵ
                        //�������֮������������������浽Pas[j][18]�������ڷ�Χ������whileѭ��
                        //�����֮��Pas[j][ ]������
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
                    else//��Ԫ��Ϊ��
                    {
                        break;
                    }
                }
            }
        }

        //----------------------------------end--------------------------------
        //������ش�����Ԫ��--------------------------------------------------------



        //������Ԫ�ӣ�Ȼ���ж��Ƿ����
        sign = false;
        for (int i = 0; i < pasNum; i++)
        {
            //�ҵ���i������Ԫ�е�����
            int temp = Pas[i][Num];
            if (temp == -1)
            {
                //˵�����Ѿ�������Ϊ��Ԫ����
                continue;
            }

            //����������Ӧ����Ԫ���ǲ���Ϊ�գ���Ϊ�գ�����
            if (Act[temp][Num] == 0)
            {
                //������Ԫ��
                for (int k = 0; k < Num; k++)
                    Act[temp][k] = Pas[i][k];
                //������Ԫ������
                Pas[i][Num] = -1;
                //��־bool��Ϊtrue��˵�����ֻ������
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

    for (int i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        //ÿ�ִ���8����Ԫ�ӣ���Χ�������� i-7 �� i

        for (int j = 0; j < pasNum; j++)
        {
            //��4535������Ԫ����û�������ڴ˷�Χ�ڵ�
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1)//��Ԫ�Ӳ�Ϊ��
                {
                    //Pas[j][]��Act[��Pas[j][18]��][]�����
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }

                    //����Pas[j][18]�������ֵ
                    //�������֮������������������浽Pas[j][18]�������ڷ�Χ������whileѭ��
                    //�����֮��Pas[j][ ]������
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
                else//��Ԫ��Ϊ��
                {
                    //Pas[j][]��������Ԫ��
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];

                    Act[index][Num] = 1;//������Ԫ�ӷǿ�
                    break;
                }

            }
        }
    }


    for (int i = lieNum % 8 - 1; i >= 0; i--)
    {
        //ÿ�ִ���1����Ԫ�ӣ���Χ���������i

        for (int j = 0; j < pasNum; j++)
        {
            //��53������Ԫ����û���������i��
            while (Pas[j][Num] == i)
            {
                if (Act[i][Num] == 1)//��Ԫ�Ӳ�Ϊ��
                {
                    //Pas[j][]��Act[i][]�����
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }

                    //����Pas[j][18]�������ֵ
                    //�������֮������������������浽Pas[j][18]�������ڷ�Χ������whileѭ��
                    //�����֮��Pas[j][ ]������
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
                else//��Ԫ��Ϊ��
                {
                    //Pas[j][]��������Ԫ��
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];

                    Act[i][Num] = 1;//������Ԫ�ӷǿ�
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
    //������ش�����Ԫ��------------------------------------------------------
       //---------------------------begin-------------------------------------

    int i;
    
    //ÿ�ִ���8����Ԫ�ӣ���Χ�������� i-7 �� i
#pragma omp parallel num_threads(thread_count) 
    for (i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
#pragma omp for schedule(dynamic,20)     
        for (int j = 0; j < pasNum; j++)
        {
            //��ǰ�����Լ����̵����񡪡�������ȥ
            if (int(j % num_proc) == rank)
            {

                while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
                {
                    int index = Pas[j][Num];

                    if (Act[index][Num] == 1)//��Ԫ�Ӳ�Ϊ��
                    {
                        //****************************SIMD******************************
                        ////Pas[j][]��Act[��Pas[j][x]��][]�����
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


                        //����Pas[j][18]�������ֵ
                        //�������֮������������������浽Pas[j][18]�������ڷ�Χ������whileѭ��
                        //�����֮��Pas[j][ ]������
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
                    else//��Ԫ��Ϊ��
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
        //ÿ�ִ���1����Ԫ�ӣ���Χ���������i
#pragma omp for schedule(dynamic,20)
        for (int j = 0; j < pasNum; j++)
        {
            //��ǰ�����Լ����̵����񡪡�������ȥ
            if (int(j % num_proc) == rank)
            {
                while (Pas[j][Num] == i)
                {
                    if (Act[i][Num] == 1)//��Ԫ�Ӳ�Ϊ��
                    {
                         //****************************SIMD******************************
                        ////Pas[j][]��Act[��Pas[j][x]��][]�����
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


                        //����Pas[j][18]�������ֵ
                        //�������֮������������������浽Pas[j][18]�������ڷ�Χ������whileѭ��
                        //�����֮��Pas[j][ ]������
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
                    else//��Ԫ��Ϊ��
                    {
                        break;
                    }
                }
            }
        }
    }
    //----------------------------------end--------------------------------
    //������ش�����Ԫ��--------------------------------------------------------

}


void f_mpi()
{

    int num_proc;//������
    int rank;//ʶ����ý��̵�rank��ֵ��0~size-1

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //0�Ž��̡������񻮷�
    if (rank == 0)
    {
        timeval t_start;
        timeval t_end;

        gettimeofday(&t_start, NULL);
        int sign;
        do
        {
            //���񻮷�
            for (int i = 0; i < pasNum; i++)
            {
                int flag = i % num_proc;
                if (flag == rank)
                    continue;
                else
                    MPI_Send(&Pas[i], Num + 1, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
            }
            super(rank, num_proc);
            //������0�Ž����Լ��������������������̴���֮��Ľ��
            for (int i = 0; i < pasNum; i++)
            {
                int flag = i % num_proc;
                if (flag == rank)
                    continue;
                else
                    MPI_Recv(&Pas[i], Num + 1, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            //������Ԫ�ӣ�Ȼ���ж��Ƿ����
            sign = 0;
            for (int i = 0; i < pasNum; i++)
            {
                //�ҵ���i������Ԫ�е�����
                int temp = Pas[i][Num];
                if (temp == -1)
                {
                    //˵�����Ѿ�������Ϊ��Ԫ����
                    continue;
                }

                //����������Ӧ����Ԫ���ǲ���Ϊ�գ���Ϊ�գ�����
                if (Act[temp][Num] == 0)
                {
                    //������Ԫ��
                    for (int k = 0; k < Num; k++)
                        Act[temp][k] = Pas[i][k];
                    //������Ԫ������
                    Pas[i][Num] = -1;
                    //��־��Ϊtrue��˵�����ֻ������
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

    //��������
    else
    {
        int sign;

        do
        {
            //��0�Ž����Ƚ�������
            for (int i = rank; i < pasNum; i += num_proc)
            {
                MPI_Recv(&Pas[i], Num + 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            //ִ������
            super(rank, num_proc);
            //��0�Ž����������֮�󣬽�������ص�0�Ž���
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






