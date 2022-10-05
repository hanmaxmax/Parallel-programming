#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <nmmintrin.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <immintrin.h> //AVX��AVX2
//#include <windows.h>
#include <sys/time.h>
using namespace std;

/*
unsigned int Act[23045][722] = { 0 };
unsigned int Pas[23045][722] = { 0 };

const int Num = 721;
const int pasNum = 14325;
const int lieNum = 23045;

*/
unsigned int Act[37960][1188] = { 0 };
unsigned int Pas[37960][1188] = { 0 };

const int Num = 1187;
const int pasNum = 14921;
const int lieNum = 37960;

//��Ԫ�ӳ�ʼ��
void init_A()
{
    //ÿ����Ԫ�ӵ�һ��Ϊ1λ���ڵ�λ�ã����������ڶ�ά������к�
    //���磺��Ԫ�ӣ�561��...����Act[561][]���
    unsigned int a;
    ifstream infile("act2.txt");
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
    ifstream infile("pas2.txt");
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
}



__m128 va_Pas;
__m128 va_Act;


void f_sse()
{
    for (int i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1)
                {

                    //*******************�����Ż�����***********************
                    //********
                    int k;
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
                    //*******
                    //********************�����Ż�����***********************


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
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];

                    Act[index][Num] = 1;
                    break;
                }
            }
        }
    }


    for (int i = lieNum % 8 - 1; i >= 0; i--)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] == i)
            {
                if (Act[i][Num] == 1)
                {

                    //*******************�����Ż�����***********************
                    //********
                    int k;
                    for (k = 0; k + 4 <= Num; k += 4)
                    {
                        //Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        va_Pas = _mm_loadu_ps((float*)&(Pas[j][k]));
                        va_Act = _mm_loadu_ps((float*)&(Act[i][k]));
                        va_Pas = _mm_xor_ps(va_Pas, va_Act);
                        _mm_store_ss((float*)&(Pas[j][k]), va_Pas);
                    }

                    for (; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }
                    //*******
                    //********************�����Ż�����***********************



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
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];

                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }

}


__m256 va_Pas2;
__m256 va_Act2;

void f_avx256()
{
    for (int i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1)
                {

                    //*******************�����Ż�����***********************
                    //********
                    int k;
                    for (k = 0; k + 8 <= Num; k += 8)
                    {
                        //Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        va_Pas2 = _mm256_loadu_ps((float*)&(Pas[j][k]));
                        va_Act2 = _mm256_loadu_ps((float*)&(Act[index][k]));

                        va_Pas2 = _mm256_xor_ps(va_Pas2, va_Act2);
                        _mm256_storeu_ps((float*)&(Pas[j][k]), va_Pas2);
                    }

                    for (; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }
                    //*******
                    //********************�����Ż�����***********************


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
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];

                    Act[index][Num] = 1;
                    break;
                }
            }
        }
    }


    for (int i = lieNum % 8 - 1; i >= 0; i--)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] == i)
            {
                if (Act[i][Num] == 1)
                {

                    //*******************�����Ż�����***********************
                    //********
                    int k;
                    for (k = 0; k + 8 <= Num; k += 8)
                    {
                        //Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        va_Pas2 = _mm256_loadu_ps((float*)&(Pas[j][k]));
                        va_Act2 = _mm256_loadu_ps((float*)&(Act[i][k]));
                        va_Pas2 = _mm256_xor_ps(va_Pas2, va_Act2);
                        _mm256_storeu_ps((float*)&(Pas[j][k]), va_Pas2);
                    }

                    for (; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }
                    //*******
                    //********************�����Ż�����***********************



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
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];

                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }
}


__m512 va_Pas3;
__m512 va_Act3;

void f_avx512()
{
    for (int i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1)
                {

                    //*******************�����Ż�����***********************
                    //********
                    int k;
                    for (k = 0; k + 16 <= Num; k += 16)
                    {
                        //Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        va_Pas3 = _mm512_loadu_ps((float*)&(Pas[j][k]));
                        va_Act3 = _mm512_loadu_ps((float*)&(Act[index][k]));

                        va_Pas3 = _mm512_xor_ps(va_Pas3, va_Act3);
                        _mm512_storeu_ps((float*)&(Pas[j][k]), va_Pas3);
                    }

                    for (; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }
                    //*******
                    //********************�����Ż�����***********************


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
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];

                    Act[index][Num] = 1;
                    break;
                }
            }
        }
    }


    for (int i = lieNum % 8 - 1; i >= 0; i--)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] == i)
            {
                if (Act[i][Num] == 1)
                {

                    //*******************�����Ż�����***********************
                    //********
                    int k;
                    for (k = 0; k + 16 <= Num; k += 16)
                    {
                        //Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        va_Pas3 = _mm512_loadu_ps((float*)&(Pas[j][k]));
                        va_Act3 = _mm512_loadu_ps((float*)&(Act[i][k]));
                        va_Pas3 = _mm512_xor_ps(va_Pas3, va_Act3);
                        _mm512_storeu_ps((float*)&(Pas[j][k]), va_Pas3);
                    }

                    for (; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }
                    //*******
                    //********************�����Ż�����***********************



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
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];

                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }
}






int main()
{
    /*
    init_A();
    init_P();
    double seconds;
    long long head, tail, freq, noww;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);//��ʼ��ʱ
    //f_ordinary();
    f_avx256();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);//������ʱ
    seconds = (tail - head) * 1000.0 / freq;//��λ ms
    cout << seconds << 'ms';
    */

    struct timeval head,tail;
    double seconds;

    init_A();
    init_P();
    gettimeofday(&head, NULL);//��ʼ��ʱ
    f_sse();
    gettimeofday(&tail, NULL);//������ʱ
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//��λ ms
    cout<<"time: "<<seconds<<" ms"<<endl;


}

