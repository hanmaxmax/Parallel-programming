#include <iostream>
#include <windows.h>

using namespace std;

const long long int N=4194304;//(2^22)

long long int mysum;
double c[N];

//初始化
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
    // 多路链式
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
    //递归函数，优点是简单，缺点是递归函数调用开销较大
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
    //二重循环实现递归
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
    long long counter;// 记录次数
    double seconds ;
    long long head,tail,freq,noww;
    init(N);

    for(long long int n=2;n<=4194304;n=n*2)
    {
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);//开始计时

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
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );//结束计时
        seconds = (tail - head) * 1000.0 / freq ;//单位 ms

        //输出结果
        cout << n <<' '<< counter <<' '<< seconds<<' '<< seconds / counter << endl ;
        //cout << seconds / counter << endl ;
    }

    return 0;
}
