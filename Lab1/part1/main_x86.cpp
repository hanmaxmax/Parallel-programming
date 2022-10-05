#include <iostream>
#include <windows.h>

using namespace std;

const int N=10000;

double b[N][N],a[N],sum[N];

//初始化
void init(int n)
{
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            b[i][j] = i+j;//初始化矩阵
        }
        a[i] = i;//初始化向量
    }
}


void f_ordinary(int n)
{
    for(int i=0;i<n;i++)
    {
        sum[i]=0.0;
        for(int j=0;j<n;j++)
        {
            sum[i]+=b[j][i]*a[j];
        }
    }
}

void f_pro(int n)
{
    for (int i=0; i<n; i++)
    {
        sum[i]=0.0;
    }
    for (int j=0;j<n;j++)
    {
        for(int i=0;i<n;i++)
        {
            sum[i]+=b[j][i]*a[j];
        }
    }


}


int main()
{
    int n,step=100;
    long long counter;// 记录次数
    double seconds ;
    long long head,tail,freq,noww;
    init(N);

    for(int n=0;n<=10000;n+=step)
    {
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);//开始计时

        counter=0;
        while(true)
        {
            QueryPerformanceCounter((LARGE_INTEGER *)&noww);
            if( (noww-head)*1000.0/freq > 10)
                break;
            // f_ordinary(n);//执行函数
            f_pro(n);
            counter++;
        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );//结束计时
        seconds = (tail - head) * 1000.0 / freq ;//单位 ms

        //输出结果
        cout << n <<' '<< counter <<' '<< seconds<<' '<< seconds / counter << endl ;
       if(n==1000)
          step=1000;
    }

    return 0;
}
