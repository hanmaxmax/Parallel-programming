#include <iostream>
#include <windows.h>

using namespace std;

const long long int N=4194304;//(2^22)

long long int mysum;
double c[N];

//��ʼ��
void init(int n)
{
    for(long long int i=0;i<N;i++)
    {
        c[i]=i-500000;
    }
}

void g_ordinary(int n)
{
    for(int i=0;i<n;i++)
    {
        mysum+=c[i];
    }
}

void g_my(int n)
{
    for(int i=0;i<n;i+=2)
    {
        mysum+=c[i];
        mysum+=c[i+1];
    }
}


void g_pro1(int n)
{
    // ��·��ʽ
    long long int mysum1=0,mysum2=0;
    for(long long int i=0;i<n;i+=2)
    {
        mysum1+=c[i];
        mysum2+=c[i+1];
    }
    mysum=mysum1+mysum2;
}

void g_pro2(int n)
{
    //�ݹ麯�����ŵ��Ǽ򵥣�ȱ���ǵݹ麯�����ÿ����ϴ�
    if (n==1)
        return;
    else
    {
        for(int i=0;i<n/2;i++)
        {
            c[i]+=c[n-i-1];
        }
        n=n/2;
        g_pro2(n);
    }
}


void g_pro3(int n)
{
    //����ѭ��ʵ�ֵݹ�
    for(long long int m=n; m>1; m/=2)
    {
        for(long long int i=0;i<m/2;i++)
        {
            c[i]=c[i*2]+c[i*2+1];
        }
    }
}


int main()
{
    long long counter;// ��¼����
    double seconds ;
    long long head,tail,freq,noww;
    init(N);

    for(long long int n=2;n<=4194304;n=n*2)
    {
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);//��ʼ��ʱ

        counter=0;
        while(true)
        {
            QueryPerformanceCounter((LARGE_INTEGER *)&noww);
            if( (noww-head)*1000.0/freq > 10)
                break;
            //g_ordinary(n);
            //g_pro1(n);
            //g_pro2(n);
            g_pro3(n);
            counter++;
        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );//������ʱ
        seconds = (tail - head) * 1000.0 / freq ;//��λ ms

        //������
        cout << n <<' '<< counter <<' '<< seconds<<' '<< seconds / counter << endl ;
        //cout << seconds / counter << endl ;
    }

    return 0;
}
