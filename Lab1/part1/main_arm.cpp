#include <iostream>
#include <sys/time.h>

using namespace std;

const int N=10000;

double b[N][N],a[N],sum[N];

//��ʼ��
void init(int n)
{
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            b[i][j] = i+j;//��ʼ������
        }
        a[i] = i;//��ʼ������
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
    long long counter;// ��¼����
    double seconds ;
    struct timeval head, tail, noww;
    init(N);

    for(int n=0;n<=10000;n+=step)
    {
        gettimeofday(&head, NULL);//��ʼ��ʱ

        counter=0;
        while(true)
        {
            gettimeofday(&noww, NULL);
            if(((noww.tv_sec - head.tv_sec)*1000000 + (noww.tv_usec - head.tv_usec)) / 1000.0 > 10)
                break;
            //f_ordinary(n);//ִ�к���
            f_ordinary(n);
            counter++;
        }
        gettimeofday(&tail, NULL);//������ʱ
        seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//��λ ms

        //������
        cout << n <<' '<< counter <<' '<< seconds<<' '<< seconds / counter << endl ;// ��λ ms
       if(n==1000)
          step=1000;
    }

    return 0;
}
