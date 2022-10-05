# include <iostream>
# include <arm_neon.h> // use Neon
# include <pthread.h>
#include <semaphore.h>
# include <sys/time.h>
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
	int t_id; //线程 id
};

//信号量定义
sem_t sem_leader;
sem_t* sem_Divsion = new sem_t[NUM_THREADS - 1]; // 每个线程有自己专属的信号量
sem_t* sem_Elimination = new sem_t[NUM_THREADS - 1];



//线程函数定义（串行）
void* threadFunc(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < n; ++k)
	{
		if (t_id == 0)
		{
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] * 1.0 / A[k][k];
			}
			A[k][k] = 1.0;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}

		// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Divsion[i]);
		}


		//循环划分任务
		for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
		{
			//消去
			for (int j = k + 1; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;

		}


		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_wait(&sem_leader); // 等待其它 worker 完成消去

			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
		}
		else
		{
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
		}
	}
	pthread_exit(NULL);
}




//线程函数定义（穿插）
void* threadFunc_horizontal1(void* param)
{
	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);
	float32x4_t vt = vmovq_n_f32(0);

	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < n; ++k)
	{

		vt = vmovq_n_f32(A[k][k]);

		if (t_id == 0)
		{
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				va = vld1q_f32(&(A[k][j]));
				va = vdivq_f32(va, vt);
				vst1q_f32(&(A[k][j]), va);
			}

			for (; j < n; j++)
			{
				A[k][j] = A[k][j] * 1.0 / A[k][k];
			}
			A[k][k] = 1.0;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}

		// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Divsion[i]);
		}


		//循环划分任务
		for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
		{
			//消去
			vaik = vmovq_n_f32(A[i][k]);
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				vakj = vld1q_f32(&(A[k][j]));
				vaij = vld1q_f32(&(A[i][j]));
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32(&A[i][j], vaij);
			}
			for (; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];

			A[i][k] = 0;
		}


		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_wait(&sem_leader); // 等待其它 worker 完成消去

			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
		}
		else
		{
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
		}
	}
	pthread_exit(NULL);
}

//水平块划分
void* threadFunc_horizontal2(void* param)
{
	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);
	float32x4_t vt = vmovq_n_f32(0);

	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < n; ++k)
	{

		vt = vmovq_n_f32(A[k][k]);

		if (t_id == 0)
		{
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				va = vld1q_f32(&(A[k][j]));
				va = vdivq_f32(va, vt);
				vst1q_f32(&(A[k][j]), va);
			}

			for (; j < n; j++)
			{
				A[k][j] = A[k][j] * 1.0 / A[k][k];
			}
			A[k][k] = 1.0;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}

		// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Divsion[i]);
		}



		int each = (n - k - 1) / NUM_THREADS;
		int end = 0;
		if (t_id == NUM_THREADS - 1)
		{
			end = n;
		}
		else
		{
			end = k + 1 + each * (t_id + 1);
		}

		//循环划分任务
		for (int i = k + 1 + t_id * each; i < end; i ++)
		{
			//消去
			vaik = vmovq_n_f32(A[i][k]);
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				vakj = vld1q_f32(&(A[k][j]));
				vaij = vld1q_f32(&(A[i][j]));
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32(&A[i][j], vaij);
			}
			for (; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];

			A[i][k] = 0;
		}


		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_wait(&sem_leader); // 等待其它 worker 完成消去

			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
		}
		else
		{
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
		}
	}
	pthread_exit(NULL);
}




//垂直块划分
void* threadFunc_vertical1(void* param)
{
	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);
	float32x4_t vt = vmovq_n_f32(0);

	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < n; ++k)
	{

		vt = vmovq_n_f32(A[k][k]);

		if (t_id == 0)
		{
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				va = vld1q_f32(&(A[k][j]));
				va = vdivq_f32(va, vt);
				vst1q_f32(&(A[k][j]), va);
			}

			for (; j < n; j++)
			{
				A[k][j] = A[k][j] * 1.0 / A[k][k];
			}
			A[k][k] = 1.0;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}

		// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Divsion[i]);
		}


		//循环划分任务
		for (int i = k + 1; i < n; i++)
		{
			//消去
			vaik = vmovq_n_f32(A[i][k]);
			int j;

			int each = (n - k - 1) / NUM_THREADS;
			int end = 0;
			if (t_id == NUM_THREADS - 1)
			{
				end = n;
			}
			else
			{
				end = k + 1 + each * (t_id + 1);
			}


			for (j = k + 1 + t_id * each; j + 4 <= end; j += 4)
			{
				vakj = vld1q_f32(&(A[k][j]));
				vaij = vld1q_f32(&(A[i][j]));
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32(&A[i][j], vaij);
			}
			for (; j < end; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];

			A[i][k] = 0;
		}


		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_wait(&sem_leader); // 等待其它 worker 完成消去

			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
		}
		else
		{
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
		}
	}
	pthread_exit(NULL);
}


//垂直穿插划分
void* threadFunc_vertica2(void* param)
{

	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < n; ++k)
	{

		if (t_id == 0)
		{
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] * 1.0 / A[k][k];
			}
			A[k][k] = 1.0;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}

		// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Divsion[i]);
		}


		//循环划分任务
		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1 + t_id; j < n; j += NUM_THREADS)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}


		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_wait(&sem_leader); // 等待其它 worker 完成消去

			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
		}
		else
		{
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
		}
	}
	pthread_exit(NULL);
	return NULL;
}














int main()
{
	init();
	struct timeval head, tail;
	double seconds;
	gettimeofday(&head, NULL);//开始计时

	//初始化信号量
	sem_init(&sem_leader, 0, 0);

	for (int i = 0; i < NUM_THREADS - 1; ++i)
	{
		sem_init(sem_Divsion, 0, 0);
		sem_init(sem_Elimination, 0, 0);
	}

	//创建线程
	pthread_t* handles = new pthread_t[NUM_THREADS];// 创建对应的 Handle
	threadParam_t* param = new threadParam_t[NUM_THREADS];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc_horizontal2, (void*)&param[t_id]);
	}


	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);

	//销毁所有信号量
	sem_destroy(&sem_leader);
	sem_destroy(sem_Divsion);
	sem_destroy(sem_Elimination);

	gettimeofday(&tail, NULL);//结束计时
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
	cout << "pthread_neon_3: " << seconds << " ms" << endl;





}
