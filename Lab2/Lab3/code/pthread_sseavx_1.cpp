# include <iostream>
# include <pthread.h>
#include <windows.h>

#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h> //AVX��AVX2
using namespace std;

const int n = 1000;
float A[n][n];
int worker_count = 7; //�����߳�����
void init()
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			A[i][j] = 0;
		}
		A[i][i] = 1.0;
		for (int j = i + 1; j < n; j++)
			A[i][j] = rand() % 100;
	}

	for (int i = 0; i < n; i++)
	{
		int k1 = rand() % n;
		int k2 = rand() % n;
		for (int j = 0; j < n; j++)
		{
			A[i][j] += A[0][j];
			A[k1][j] += A[k2][j];
		}
	}
}

void f_ordinary()
{
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j++)
		{
			A[k][j] = A[k][j] * 1.0 / A[k][k];
		}
		A[k][k] = 1.0;

		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
}


struct threadParam_t
{
	int k; //��ȥ���ִ�
	int t_id; // �߳� id
};

void* threadFunc(void* param)
{

   __m256 va, vt, vx, vaij, vaik, vakj;

	threadParam_t* p = (threadParam_t*)param;
	int k = p->k; //��ȥ���ִ�
	int t_id = p->t_id; //�̱߳��
	int i = k + t_id + 1; //��ȡ�Լ��ļ�������
	for (int m = k + 1 + t_id; m < n; m += worker_count)
	{
	    vaik = _mm256_set_ps(A[m][k], A[m][k], A[m][k], A[m][k],A[m][k], A[m][k], A[m][k], A[m][k]);
        int j;
        for (j = k + 1; j+8 <= n; j+=8)
        {
            vakj = _mm256_loadu_ps(&(A[k][j]));
            vaij = _mm256_loadu_ps(&(A[m][j]));
            vx = _mm256_mul_ps(vakj, vaik);
            vaij = _mm256_sub_ps(vaij, vx);

            _mm256_store_ps(&A[i][j], vaij);
        }
        for(; j<n; j++)
            A[m][j] = A[m][j] - A[m][k] * A[k][j];

        A[m][k] = 0;
	}


	pthread_exit(NULL);

}


int main()
{
	init();
    __m256 va2, vt2, vx2, vaij2, vaik2, vakj2;

    long long counter;// ��¼����
    double seconds ;
    long long head,tail,freq,noww;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);//��ʼ��ʱ


	for (int k = 0; k < n; k++)
	{
		vt2 = _mm256_set_ps(A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]);
	    int j;
		for (j = k + 1; j+8 <= n; j+=8)
		{
		    va2 = _mm256_loadu_ps(&(A[k][j]));
			va2 = _mm256_div_ps(va2, vt2);
			_mm256_store_ps(&(A[k][j]), va2);
		}

		for(; j<n; j++)
        {
            A[k][j]=A[k][j]*1.0 / A[k][k];

        }
		A[k][k] = 1.0;

		//���������̣߳�������ȥ����

		pthread_t* handles = new pthread_t[worker_count];// ������Ӧ�� Handle
		threadParam_t* param = new threadParam_t[worker_count];// ������Ӧ���߳����ݽṹ

		//��������
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//�����߳�
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);

		//���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_join(handles[t_id], NULL);

	}


	QueryPerformanceCounter((LARGE_INTEGER *)&tail );//������ʱ
    seconds = (tail - head) * 1000.0 / freq ;//��λ ms

    cout<<"pthread_dong: "<<seconds<<" ms"<<endl;

}
