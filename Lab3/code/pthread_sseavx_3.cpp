# include <iostream>
# include <pthread.h>
#include <semaphore.h>
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
int NUM_THREADS = 7;
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
	int t_id; //�߳� id
};

//barrier ����
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;


//�̺߳�������
void* threadFunc(void* param)
{
  __m256 va2, vt2, vx2, vaij2, vaik2, vakj2;



	threadParam_t* p = (threadParam_t*)param;
	int t_id = p -> t_id;

	for (int k = 0; k < n; ++k)
	{
        vt2 = _mm256_set_ps(A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]);
		if (t_id == 0)
		{
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
		}

		//��һ��ͬ����
		pthread_barrier_wait(&barrier_Divsion);

		//ѭ����������ͬѧ�ǿ��Գ��Զ������񻮷ַ�ʽ��
		for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
		{
			//��ȥ
			vaik2 = _mm256_set_ps(A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k]);
			int j;
			for (j = k + 1; j+8 <= n; j+=8)
			{
				vakj2 = _mm256_loadu_ps(&(A[k][j]));
				vaij2 = _mm256_loadu_ps(&(A[i][j]));
				vx2 = _mm256_mul_ps(vakj2, vaik2);
				vaij2 = _mm256_sub_ps(vaij2, vx2);

				_mm256_store_ps(&A[i][j], vaij2);
			}
			for(; j<n; j++)
                A[i][j] = A[i][j] - A[i][k] * A[k][j];

			A[i][k] = 0;
		}
		// �ڶ���ͬ����
		pthread_barrier_wait(&barrier_Elimination);

	}
	pthread_exit(NULL);
}




int main()
{
	init();
	long long counter;// ��¼����
    double seconds ;
    long long head,tail,freq,noww;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);//��ʼ��ʱ


	//��ʼ��barrier
	pthread_barrier_init(&barrier_Divsion, NULL, NUM_THREADS);
	pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);


	//�����߳�
	pthread_t* handles = new pthread_t[NUM_THREADS];// ������Ӧ�� Handle
	threadParam_t* param = new threadParam_t[NUM_THREADS];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
	}


	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);

	//�������е� barrier
	pthread_barrier_destroy(&barrier_Divsion);
	pthread_barrier_destroy(&barrier_Elimination);



	QueryPerformanceCounter((LARGE_INTEGER *)&tail );//������ʱ
    seconds = (tail - head) * 1000.0 / freq ;//��λ ms

    cout<<"pthread_neon_4: "<<seconds<<" ms"<<endl;


}
