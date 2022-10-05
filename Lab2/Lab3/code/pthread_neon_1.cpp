# include <iostream>
# include <pthread.h>
# include <sys/time.h>
# include <arm_neon.h> // use Neon
using namespace std;

const int n = 1000;
float A[n][n];
int worker_count = 7; //工作线程数量
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
	int k; //消去的轮次
	int t_id; // 线程 id
};

void* threadFunc(void* param)
{

    float32x4_t vx = vmovq_n_f32(0);
    float32x4_t vaij = vmovq_n_f32(0);
    float32x4_t vaik = vmovq_n_f32(0);
    float32x4_t vakj = vmovq_n_f32(0);


	threadParam_t* p = (threadParam_t*)param;
	int k = p->k; //消去的轮次
	int t_id = p->t_id; //线程编号
	int i = k + t_id + 1; //获取自己的计算任务
	for (int m = k + 1 + t_id; m < n; m += worker_count)
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
            A[m][j] = A[m][j] - A[m][k] * A[k][j];

        A[m][k] = 0;
	}


	pthread_exit(NULL);

}


int main()
{
	init();
    struct timeval head,tail;
    double seconds;
    gettimeofday(&head, NULL);//开始计时


    float32x4_t va = vmovq_n_f32(0);
    float32x4_t vt = vmovq_n_f32(0);

	for (int k = 0; k < n; k++)
	{
		vt=vmovq_n_f32(A[k][k]);
	    int j;
		for (j = k + 1; j+4 <= n; j+=4)
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

		//创建工作线程，进行消去操作

		pthread_t* handles = new pthread_t[worker_count];// 创建对应的 Handle
		threadParam_t* param = new threadParam_t[worker_count];// 创建对应的线程数据结构

		//分配任务
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//创建线程
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);

		//主线程挂起等待所有的工作线程完成此轮消去工作
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_join(handles[t_id], NULL);

	}


	gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"pthread_neon_1: "<<seconds<<" ms"<<endl;

}
