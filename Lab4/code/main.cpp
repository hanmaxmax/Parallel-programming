#include <omp.h>
#include <iostream>
#include <windows.h>
using namespace std;

const int n = 1000;
float arr[n][n];
float A[n][n];
const int NUM_THREADS = 7; //工作线程数量


void init()
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			arr[i][j] = 0;
		}
		arr[i][i] = 1.0;
		for (int j = i + 1; j < n; j++)
			arr[i][j] = rand() % 100;
	}

	for (int i = 0; i < n; i++)
	{
		int k1 = rand() % n;
		int k2 = rand() % n;
		for (int j = 0; j < n; j++)
		{
			arr[i][j] += arr[0][j];
			arr[k1][j] += arr[k2][j];
		}
	}
}


void ReStart()
{
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
            A[i][j]=arr[i][j];
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

void f_omp_static()
{
	 #pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < n; k++)
	{
		//串行部分
		#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//并行部分
		#pragma omp for schedule(static)
		for (int i = k + 1; i < n; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}


void f_omp_dynamic()
{
	 #pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < n; k++)
	{
		//串行部分
		#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//并行部分
		#pragma omp for schedule(dynamic, 80)
		for (int i = k + 1; i < n; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}



void f_omp_guided()
{
	 #pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < n; k++)
	{
		//串行部分
		#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//并行部分
		#pragma omp for schedule(guided, 80)
		for (int i = k + 1; i < n; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}










/*

void f_omp_static_neon()
{
    float32x4_t va = vmovq_n_f32(0);
    float32x4_t vx = vmovq_n_f32(0);
    float32x4_t vaij = vmovq_n_f32(0);
    float32x4_t vaik = vmovq_n_f32(0);
    float32x4_t vakj = vmovq_n_f32(0);

    #pragma omp parallel num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj)
	for (int k = 0; k < n; k++)
	{
		//串行部分
		#pragma omp single
		{
		    float32x4_t vt=vmovq_n_f32(A[k][k]);
            int j;
			for (j = k + 1; j < n; j++)
			{
				va=vld1q_f32(&(A[k][j]) );
                va= vdivq_f32(va,vt);
                vst1q_f32(&(A[k][j]), va);
			}
			for(; j<n; j++)
            {
                A[k][j]=A[k][j]*1.0 / A[k][k];

            }
            A[k][k] = 1.0;
		}

		//并行部分
		#pragma omp for schedule(static)
		for (int i = k + 1; i < n; i++)
		{
		    vaik=vmovq_n_f32(A[i][k]);
            int j;
			for (j = k + 1; j+4 <= n; j+=4)
			{
				vakj=vld1q_f32(&(A[k][j]));
				vaij=vld1q_f32(&(A[i][j]));
				vx=vmulq_f32(vakj,vaik);
				vaij=vsubq_f32(vaij,vx);

				vst1q_f32(&A[i][j], vaij);
			}

			for(; j<n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }

			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}





void f_omp_dynamic_neon()
{
    float32x4_t va = vmovq_n_f32(0);
    float32x4_t vx = vmovq_n_f32(0);
    float32x4_t vaij = vmovq_n_f32(0);
    float32x4_t vaik = vmovq_n_f32(0);
    float32x4_t vakj = vmovq_n_f32(0);

    #pragma omp parallel num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj)
	for (int k = 0; k < n; k++)
	{
		//串行部分
		#pragma omp single
		{
		    float32x4_t vt=vmovq_n_f32(A[k][k]);
            int j;
			for (j = k + 1; j < n; j++)
			{
				va=vld1q_f32(&(A[k][j]) );
                va= vdivq_f32(va,vt);
                vst1q_f32(&(A[k][j]), va);
			}
			for(; j<n; j++)
            {
                A[k][j]=A[k][j]*1.0 / A[k][k];

            }
            A[k][k] = 1.0;
		}

		//并行部分
		#pragma omp for schedule(dynamic, 5)
		for (int i = k + 1; i < n; i++)
		{
		    vaik=vmovq_n_f32(A[i][k]);
            int j;
			for (j = k + 1; j+4 <= n; j+=4)
			{
				vakj=vld1q_f32(&(A[k][j]));
				vaij=vld1q_f32(&(A[i][j]));
				vx=vmulq_f32(vakj,vaik);
				vaij=vsubq_f32(vaij,vx);

				vst1q_f32(&A[i][j], vaij);
			}

			for(; j<n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }

			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}



void f_omp_guided_neon()
{
    float32x4_t va = vmovq_n_f32(0);
    float32x4_t vx = vmovq_n_f32(0);
    float32x4_t vaij = vmovq_n_f32(0);
    float32x4_t vaik = vmovq_n_f32(0);
    float32x4_t vakj = vmovq_n_f32(0);

    #pragma omp parallel num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj)
	for (int k = 0; k < n; k++)
	{
		//串行部分
		#pragma omp single
		{
		    float32x4_t vt=vmovq_n_f32(A[k][k]);
            int j;
			for (j = k + 1; j < n; j++)
			{
				va=vld1q_f32(&(A[k][j]) );
                va= vdivq_f32(va,vt);
                vst1q_f32(&(A[k][j]), va);
			}
			for(; j<n; j++)
            {
                A[k][j]=A[k][j]*1.0 / A[k][k];

            }
            A[k][k] = 1.0;
		}

		//并行部分
		#pragma omp for schedule(guided, 5)
		for (int i = k + 1; i < n; i++)
		{
		    vaik=vmovq_n_f32(A[i][k]);
            int j;
			for (j = k + 1; j+4 <= n; j+=4)
			{
				vakj=vld1q_f32(&(A[k][j]));
				vaij=vld1q_f32(&(A[i][j]));
				vx=vmulq_f32(vakj,vaik);
				vaij=vsubq_f32(vaij,vx);

				vst1q_f32(&A[i][j], vaij);
			}

			for(; j<n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }

			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}


*/




int main()
{
	init();
    double seconds ;
    long long head,tail,freq,noww;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);



	ReStart();
	QueryPerformanceCounter((LARGE_INTEGER *)&head);//开始计时
	f_ordinary();
	QueryPerformanceCounter((LARGE_INTEGER *)&tail );//结束计时
    seconds = (tail - head) * 1000.0 / freq ;//单位 ms
	cout << "f_ordinary: " << seconds << " ms" << endl;



	ReStart();
	QueryPerformanceCounter((LARGE_INTEGER *)&head);//开始计时
	f_omp_static();
	QueryPerformanceCounter((LARGE_INTEGER *)&tail );//结束计时
	seconds = (tail - head) * 1000.0 / freq ;//单位 ms
	cout << "f_omp_static: " << seconds << " ms" << endl;



	ReStart();
	QueryPerformanceCounter((LARGE_INTEGER *)&head);//开始计时
	f_omp_dynamic();
	QueryPerformanceCounter((LARGE_INTEGER *)&tail );//结束计时
	seconds = (tail - head) * 1000.0 / freq ;//单位 ms
	cout << "f_omp_dynamic: " << seconds << " ms" << endl;


	ReStart();
	QueryPerformanceCounter((LARGE_INTEGER *)&head);//开始计时
	f_omp_guided();
	QueryPerformanceCounter((LARGE_INTEGER *)&tail );//结束计时
	seconds = (tail - head) * 1000.0 / freq ;//单位 ms
	cout << "f_omp_guided: " << seconds << " ms" << endl;


	/*
	ReStart();
	QueryPerformanceCounter((LARGE_INTEGER *)&head);//开始计时
	f_omp_static_neon();
	QueryPerformanceCounter((LARGE_INTEGER *)&tail );//结束计时
	seconds = (tail - head) * 1000.0 / freq ;//单位 ms
	cout << "f_omp_static_neon: " << seconds << " ms" << endl;



	ReStart();
	QueryPerformanceCounter((LARGE_INTEGER *)&head);//开始计时
	f_omp_dynamic_neon();
	QueryPerformanceCounter((LARGE_INTEGER *)&tail );//结束计时
	seconds = (tail - head) * 1000.0 / freq ;//单位 ms
	cout << "f_omp_dynamic_neon: " << seconds << " ms" << endl;


	ReStart();
	QueryPerformanceCounter((LARGE_INTEGER *)&head);//开始计时
	f_omp_guided_neon();
	QueryPerformanceCounter((LARGE_INTEGER *)&tail );//结束计时
	seconds = (tail - head) * 1000.0 / freq ;//单位 ms
	cout << "f_omp_guided_neon: " << seconds << " ms" << endl;
*/

}




