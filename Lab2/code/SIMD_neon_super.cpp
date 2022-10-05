// �����˹��ȥ��neon����
# include <arm_neon.h> // use Neon
# include <sys/time.h>

#include <iostream>
#include <sstream>
#include <fstream>
using namespace std;

unsigned int Act[37960][1188] = { 0 };
unsigned int Pas[37960][1188] = { 0 };

//��Ԫ�ӳ�ʼ��
void init_A()
{
    //ÿ����Ԫ�ӵ�һ��Ϊ1λ���ڵ�λ�ã����������ڶ�ά������к�
    //���磺��Ԫ�ӣ�561��...����Act[561][]���
    unsigned int a;
    ifstream infile("act.txt");
    char fin[100000] = { 0 };
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
            Act[index][1186 - j] += temp;
            Act[index][1187] = 1;//���ø�λ�ü�¼��Ԫ�Ӹ����Ƿ�Ϊ�գ�Ϊ������0������Ϊ1
        }
    }
}

//����Ԫ�г�ʼ��
void init_P()
{
    //ֱ�Ӱ��մ����ļ���˳��棬�ڴ����ļ��ǵڼ��У���������ǵڼ���
    unsigned int a;
    ifstream infile("pas.txt");
    char fin[100000] = { 0 };
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
                Pas[index][1187] = a;
                biaoji = 1;
            }

            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Pas[index][1186 - j] += temp;
        }
        index++;
    }
}





void f_ordinary()
{
    int i;
    for (i = 37959; i - 8 >= -1; i -= 8)
    {
        //ÿ�ִ���8����Ԫ�ӣ���Χ�������� i-7 �� i

        for (int j = 0; j < 14921; j++)
        {
            //��4535������Ԫ����û�������ڴ˷�Χ�ڵ�
            while (Pas[j][1187] <= i && Pas[j][1187] >= i - 7)
            {
                int index = Pas[j][1187];
                if (Act[index][1187] == 1)//��Ԫ�Ӳ�Ϊ��
                {
                    //Pas[j][]��Act[��Pas[j][18]��][]�����
                    for (int k = 0; k < 1187; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }

                    //����Pas[j][18]�������ֵ
                    //�������֮������������������浽Pas[j][18]�������ڷ�Χ������whileѭ��
                    //�����֮��Pas[j][ ]������
                    int num = 0, S_num = 0;
                    for (num = 0; num < 1187; num++)
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
                    Pas[j][1187] = S_num - 1;

                }
                else//��Ԫ��Ϊ��
                {
                    //Pas[j][]��������Ԫ��
                    for (int k = 0; k < 1187; k++)
                        Act[index][k] = Pas[j][k];

                    Act[index][1187] = 1;//������Ԫ�ӷǿ�
                    break;
                }

            }
        }
    }


    for (i = i + 8; i >= 0; i--)
    {
        //ÿ�ִ���1����Ԫ�ӣ���Χ���������i

        for (int j = 0; j < 14921; j++)
        {
            //��53������Ԫ����û���������i��
            while (Pas[j][1187] == i)
            {
                if (Act[i][1187] == 1)//��Ԫ�Ӳ�Ϊ��
                {
                    //Pas[j][]��Act[i][]�����
                    for (int k = 0; k < 1187; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }

                    //����Pas[j][18]�������ֵ
                    //�������֮������������������浽Pas[j][18]�������ڷ�Χ������whileѭ��
                    //�����֮��Pas[j][ ]������
                    int num = 0, S_num = 0;
                    for (num = 0; num < 1187; num++)
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
                    Pas[j][1187] = S_num - 1;

                }
                else//��Ԫ��Ϊ��
                {
                    //Pas[j][]��������Ԫ��
                    for (int k = 0; k < 1187; k++)
                        Act[i][k] = Pas[j][k];

                    Act[i][1187] = 1;//������Ԫ�ӷǿ�
                    break;
                }
            }
        }
    }
}


//neon���У�����ڴ��д�����в����Ż��Ĳ����Ѿ���ע�ͱ�����ˣ�
void f_pro()
{
    int i;
    for (i = 37959; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < 14921; j++)
        {
            while (Pas[j][1187] <= i && Pas[j][1187] >= i - 7)
            {
                int index = Pas[j][1187];
                if (Act[index][1187] == 1)
                {

                    //*******************�����Ż�����***********************
                    //********
                    int k;
                    for (k = 0; k+4 <= 1187; k+=4)
                    {
                        //Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        uint32x4_t vaPas =  vld1q_u32(& (Pas[j][k]));
                        uint32x4_t vaAct =  vld1q_u32(& (Act[index][k]));

                        vaPas = veorq_u32(vaPas,vaAct);
                        vst1q_u32( &(Pas[j][k]) , vaPas );
                    }

                    for( ; k<1187; k++ )
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }
                    //*******
                    //********************�����Ż�����***********************


                    int num = 0, S_num = 0;
                    for (num = 0; num < 1187; num++)
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
                    Pas[j][1187] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < 1187; k++)
                        Act[index][k] = Pas[j][k];

                    Act[index][1187] = 1;
                    break;
                }
            }
        }
    }


    for (i = i + 8; i >= 0; i--)
    {
        for (int j = 0; j < 14921; j++)
        {
            while (Pas[j][1187] == i)
            {
                if (Act[i][1187] == 1)
                {

                    //*******************�����Ż�����***********************
                    //********
                    int k;
                    for (k = 0; k+4 <= 1187; k+=4)
                    {
                        //Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        uint32x4_t va_Pas =  vld1q_u32(& (Pas[j][k]));
                        uint32x4_t va_Act =  vld1q_u32(& (Act[i][k]));

                        va_Pas = veorq_u32(va_Pas,va_Act);
                        vst1q_u32( &(Pas[j][k]) , va_Pas );
                    }

                    for( ; k<1187; k++ )
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }
                    //*******
                    //********************�����Ż�����***********************



                    int num = 0, S_num = 0;
                    for (num = 0; num < 1187; num++)
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
                    Pas[j][1187] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < 1187; k++)
                        Act[i][k] = Pas[j][k];

                    Act[i][1187] = 1;
                    break;
                }
            }
        }
    }
}




int main()
{

    struct timeval head,tail;

    init_A();
    init_P();
    gettimeofday(&head, NULL);//��ʼ��ʱ
    f_ordinary();
    gettimeofday(&tail, NULL);//������ʱ
    double seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//��λ ms
    cout<<"f_ordinary: "<<seconds<<" ms"<<endl;

    init_A();
    init_P();
    gettimeofday(&head, NULL);//��ʼ��ʱ
    f_pro();
    gettimeofday(&tail, NULL);//������ʱ
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//��λ ms
    cout<<"f_pro: "<<seconds<<" ms"<<endl;

    //getResult();

}






