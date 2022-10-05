# include <iostream>
# include <pthread.h>
#include <semaphore.h>
# include <sys/time.h>
# include <arm_neon.h> // use Neon
using namespace std;

const int n = 4000;
float A[n][n];
int NUM_THREADS = 5;
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

//�ź�������
sem_t sem_main;
sem_t* sem_workerstart = new sem_t[NUM_THREADS]; // ÿ���߳����Լ�ר�����ź���
sem_t* sem_workerend = new sem_t[NUM_THREADS];



//�̺߳�������
void* threadFunc(void* param)
{

    float32x4_t vx = vmovq_n_f32(0);
    float32x4_t vaij = vmovq_n_f32(0);
    float32x4_t vaik = vmovq_n_f32(0);
    float32x4_t vakj = vmovq_n_f32(0);


	threadParam_t *p = (threadParam_t*)param;
	int t_id = p -> t_id;
	for (int k = 0; k < n; k++)
	{
		sem_wait(&sem_workerstart[t_id]); // �������ȴ�������ɳ��������������Լ�ר�����ź�����
		//ѭ����������
		for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
		{
			//��ȥ
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
                A[i][j] = A[i][j] - A[i][k] * A[k][j];

			A[i][k] = 0.0;
		}

		sem_post(&sem_main); // �������߳�
		sem_wait(&sem_workerend[t_id]); //�������ȴ����̻߳��ѽ�����һ��
	}

	pthread_exit(NULL);

}




int main()
{
	init();
	struct timeval head,tail;
    double seconds;
    gettimeofday(&head, NULL);//��ʼ��ʱ

	//��ʼ���ź���
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < NUM_THREADS; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}

	//�����߳�
	pthread_t* handles = new pthread_t[NUM_THREADS];// ������Ӧ�� Handle
	threadParam_t* param=new threadParam_t[NUM_THREADS];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
	}


	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vt = vmovq_n_f32(0);

	for (int k = 0; k < n; ++k)
	{
		//���߳�����������
		vt=vmovq_n_f32(A[k][k]);
	    int j;
		for (j = k + 1; j+4 <= n; j+=4)
		{
		    va=vld1q_f32(&(A[k][j]) );
			va= vdivq_f32(va,vt);
			vst1q_f32(&(A[k][j]), va);
		}

		for(; j<n; j++)
            A[k][j]=A[k][j]*1.0 / A[k][k];

		A[k][k] = 1.0;


		//��ʼ���ѹ����߳�
		for(int t_id = 0; t_id < NUM_THREADS; ++t_id)
			sem_post(&sem_workerstart[t_id]);

		//���߳�˯�ߣ��ȴ����еĹ����߳���ɴ�����ȥ����
		for(int t_id = 0; t_id < NUM_THREADS; ++t_id)
			sem_wait(&sem_main);

		// ���߳��ٴλ��ѹ����߳̽�����һ�ִε���ȥ����
		for(int t_id = 0; t_id < NUM_THREADS; ++t_id)
			sem_post(&sem_workerend[t_id]);

	}

	for(int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);

	//���������ź���
	sem_destroy(&sem_main);
	sem_destroy(sem_workerstart);
	sem_destroy(sem_workerend);




	gettimeofday(&tail, NULL);//������ʱ
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//��λ ms
    cout<<"pthread_neon_2: "<<seconds<<" ms"<<endl;


}

