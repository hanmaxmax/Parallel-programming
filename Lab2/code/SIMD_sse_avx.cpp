#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <nmmintrin.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <immintrin.h> //AVX、AVX2
//#include <windows.h>
#include <sys/time.h>
using namespace std;

const int n = 300;
float A[n][n];


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
			A[i][j] = rand();
	}

	for (int k = 0; k < n; k++)
	{
		for (int i = k + 1; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				A[i][j] += A[k][j];
			}
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


__m128 va, vt, vx, vaij, vaik, vakj;
void f_sse()
{
	for (int k = 0; k < n; k++)
	{
		vt = _mm_set_ps(A[k][k], A[k][k], A[k][k], A[k][k]);
		int j;
		for (j = k + 1; j + 4 <= n; j += 4)
		{
			va = _mm_loadu_ps(&(A[k][j]));
			va = _mm_div_ps(va, vt);
			_mm_store_ps(&(A[k][j]), va);
		}

		for (; j < n; j++)
		{
			A[k][j] = A[k][j] * 1.0 / A[k][k];

		}
		A[k][k] = 1.0;

		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm_set_ps(A[i][k], A[i][k], A[i][k], A[i][k]);

			for (j = k + 1; j + 4 <= n; j += 4)
			{
				vakj = _mm_loadu_ps(&(A[k][j]));
				vaij = _mm_loadu_ps(&(A[i][j]));
				vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);

				_mm_store_ps(&A[i][j], vaij);
			}

			for (; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}

			A[i][k] = 0;
		}
	}

}



__m256 va2, vt2, vx2, vaij2, vaik2, vakj2;

void f_avx256()
{
	for (int k = 0; k < n; k++)
	{
		vt2 = _mm256_set_ps(A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]);
		int j;
		for (j = k + 1; j + 8 <= n; j += 8)
		{
			va2 = _mm256_loadu_ps(&(A[k][j]));
			va2 = _mm256_div_ps(va2, vt2);
			_mm256_store_ps(&(A[k][j]), va2);
		}

		for (; j < n; j++)
		{
			A[k][j] = A[k][j] * 1.0 / A[k][k];

		}
		A[k][k] = 1.0;

		for (int i = k + 1; i < n; i++)
		{
			vaik2 = _mm256_set_ps(A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k]);

			for (j = k + 1; j + 8 <= n; j += 8)
			{
				vakj2 = _mm256_loadu_ps(&(A[k][j]));
				vaij2 = _mm256_loadu_ps(&(A[i][j]));
				vx2 = _mm256_mul_ps(vakj2, vaik2);
				vaij2 = _mm256_sub_ps(vaij2, vx2);

				_mm256_store_ps(&A[i][j], vaij2);
			}

			for (; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}

			A[i][k] = 0;
		}
	}
}


__m512 va3, vt3, vx3, vaij3, vaik3, vakj3;

void f_avx512()
{
	for (int k = 0; k < n; k++)
	{
		float temp[16] = { A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k] };
		vt3 = _mm512_loadu_ps(temp);

		//vt3 = _mm512_set_ps(A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]);

		int j;
		for (j = k + 1; j + 16 <= n; j += 16)
		{
			va3 = _mm512_loadu_ps(&(A[k][j]));
			va3 = _mm512_div_ps(va3, vt3);
			_mm512_store_ps(&(A[k][j]), va3);
		}

		for (; j < n; j++)
		{
			A[k][j] = A[k][j] * 1.0 / A[k][k];

		}
		A[k][k] = 1.0;

		for (int i = k + 1; i < n; i++)
		{
			vaik3 = _mm512_set_ps(A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k]);

			for (j = k + 1; j + 16 <= n; j += 16)
			{
				vakj3 = _mm512_loadu_ps(&(A[k][j]));
				vaij3 = _mm512_loadu_ps(&(A[i][j]));
				vx3 = _mm512_mul_ps(vakj3, vaik3);
				vaij3 = _mm512_sub_ps(vaij3, vx3);

				_mm512_store_ps(&A[i][j], vaij3);
			}

			for (; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}

			A[i][k] = 0;
		}
	}
}




int main()
{

    /*
	double seconds;
	long long head, tail, freq, noww;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	init();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
	f_ordinary();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
	seconds = (tail - head) * 1000.0 / freq;//单位 ms
	cout << seconds << 'ms' << endl;


	init();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
	f_sse();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
	seconds = (tail - head) * 1000.0 / freq;//单位 ms
	cout << seconds << 'ms' << endl;


	init();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
	f_avx256();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
	seconds = (tail - head) * 1000.0 / freq;//单位 ms
	cout << seconds << 'ms' << endl;

	init();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
	//f_avx512();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
	seconds = (tail - head) * 1000.0 / freq;//单位 ms
	cout << seconds << 'ms' << endl;
    */


    struct timeval head,tail;
    double seconds;


    init();
    gettimeofday(&head, NULL);//开始计时
    f_ordinary();
    gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_ordinary: "<<seconds<<" ms"<<endl;


    init();
    gettimeofday(&head, NULL);//开始计时
    f_sse();
    gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_sse: "<<seconds<<" ms"<<endl;

    init();
    gettimeofday(&head, NULL);//开始计时
    f_avx256();
    gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_avx256: "<<seconds<<" ms"<<endl;

    init();
    gettimeofday(&head, NULL);//开始计时
    f_avx512();
    gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_avx512: "<<seconds<<" ms"<<endl;


}






