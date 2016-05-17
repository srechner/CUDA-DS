/*
 * mystack.h
 *
 *  Created on: May 15, 2016
 *      Author: rechner
 */

#ifndef MYSTACK_CUH_
#define MYSTACK_CUH_

#include <cuda_runtime.h>

/**
 * A stack data structure that can be used within kernels. Each thread has its own
 * stack. 
 * 
 * @param T: Generic class type.
 * @param threadsPerBlock: The number of threads in a cuda block.
 * @param sharedMemory:  The size of the shared memory that can be used for the stacks.
 */
template<class T, int threadsPerBlock, int sharedMemory>
class stack {

private:

	/**
	 *  shared memory is divided into five parts:
	 *
	 *  extern __shared__ char shared[];
	 *	Segment** const next 	= (Segment**)	&shared[0];
	 *  int* const 		pos 	= (int*) 		&shared[threadsPerBlock*sizeof(Segment*)];
	 *	int* const 		count 	= (int*) 		&shared[threadsPerBlock*(sizeof(Segment*)+sizeof(int))];
	 *	int* const 		size 	= (int*) 		&shared[threadsPerBlock*(sizeof(Segment*)+2*sizeof(int))];
	 *	T* const 		values	= (T*) 			&shared[threadsPerBlock*(sizeof(Segment*)+3*sizeof(int))];
	 */

	struct Segment;

	/**
	 * derived static variables:
	 * 
	 * N: half the number of cached stack elements per thread.
	 * M: M*N is the number of stack elements in each segment
	 * 
	 */
	static const int N = (sharedMemory
			- (sizeof(Segment*) + 3*sizeof(int))
					* threadsPerBlock) / (threadsPerBlock * 2 * sizeof(T));
	static const int M = 128;

	/**
	 * A stripe of memory where swapped stack elements are stored.
	 */
	struct Segment {

	public:
		T data[N * M];
		Segment* next;

		__device__
		Segment() {
			next = nullptr;
		}

	};

	__device__ __forceinline__
	static int index(const int i, const int id) {
		return i * threadsPerBlock + id;
	}

public:

	// init empty stack
	__device__
	static void init() {

		extern __shared__ char shared[];
		Segment** 		next 	= (Segment**)	&shared[0];
		int* 	 		pos 	= (int*) 		&shared[threadsPerBlock*sizeof(Segment*)];
		int* 	 		count 	= (int*) 		&shared[threadsPerBlock*(sizeof(Segment*)+sizeof(int))];
		int* 	 		size 	= (int*) 		&shared[threadsPerBlock*(sizeof(Segment*)+2*sizeof(int))];
		//T* const 		values	= (T*) 			&shared[threadsPerBlock*(sizeof(Segment*)+3*sizeof(int))];

		/*if (id == 0 && blockIdx.x == 0) {
			int usedShared = threadsPerBlock*(sizeof(Segment*) + 3*sizeof(int) + 2*N*sizeof(T));
			printf(
					"init stack: blockId=%i, sizeof(T)=%i, threadsPerBlock=%i, sharedMemory=%i, N=%i, usedShared=%i\n",
					blockIdx.x, (int) sizeof(T), threadsPerBlock, sharedMemory,
					N, usedShared);
		}*/

		const int id = threadIdx.x;

		count[id] = 0;
		size[id] = 0;
		pos[id] = M;
		next[id] = nullptr;
	}

	// returns true, of stack is empty, false if not
	__device__
	static bool empty() {

		extern __shared__ char shared[];
		//Segment** 		next 	= (Segment**)	&shared[0];
		//int* 	 		pos 	= (int*) 		&shared[threadsPerBlock*sizeof(Segment*)];
		//int* 	 		count 	= (int*) 		&shared[threadsPerBlock*(sizeof(Segment*)+sizeof(int))];
		int* 	 		size 	= (int*) 		&shared[threadsPerBlock*(sizeof(Segment*)+2*sizeof(int))];
		//T* 		 		values	= (T*) 			&shared[threadsPerBlock*(sizeof(Segment*)+3*sizeof(int))];

		const int id = threadIdx.x;
		return size[id] == 0;
	}

	// pushes element x onto stack
	__device__
	static void push(const T& x) {

		extern __shared__ char shared[];
		Segment** 	next 	= (Segment**)	&shared[0];
		int* 		pos 	= (int*) 		&shared[threadsPerBlock*sizeof(Segment*)];
		int* 		count 	= (int*) 		&shared[threadsPerBlock*(sizeof(Segment*)+sizeof(int))];
		int*  		size 	= (int*) 		&shared[threadsPerBlock*(sizeof(Segment*)+2*sizeof(int))];
		T*  		values	= (T*) 			&shared[threadsPerBlock*(sizeof(Segment*)+3*sizeof(int))];

		const int id = threadIdx.x;

		// swap out
		if (count[id] == 2 * N) {

	//		printf("thread %i: count=%i, pos=%i swap out\n", id, count[id], pos[id]);

			if (pos[id] == M) {

		//		printf("thread %i create new segment\n", id);

				// create new segment
				Segment* s = new Segment();
				if(s == nullptr) {
					printf("Error! not enough GPU heap space!\n");
				}
				s->next = next[id];
				next[id] = s;
				pos[id] = 0;
			}

			// copy data into new segment
			for (int i = 0; i < N; i++) {
				next[id]->data[pos[id] * N + i] = values[index(i, id)];
				values[index(i, id)] = values[index(i + N, id)];
			}

			pos[id]++;
			count[id] -= N;
		}

		values[index(count[id], id)] = x;
		count[id]++;
		size[id]++;
	}

	__device__
	static int size() {

		extern __shared__ char shared[];
		//Segment** const next 	= (Segment**)	&shared[0];
		//int* const 		pos 	= (int*) 		&shared[threadsPerBlock*sizeof(Segment*)];
		//int* const 		count 	= (int*) 		&shared[threadsPerBlock*(sizeof(Segment*)+sizeof(int))];
		int* 	 		size 	= (int*) 		&shared[threadsPerBlock*(sizeof(Segment*)+2*sizeof(int))];
		//T* const 		values	= (T*) 			&shared[threadsPerBlock*(sizeof(Segment*)+3*sizeof(int))];

		const int id = threadIdx.x;

		return size[id];
	}

	// pops upper element from stack
	__device__
	static T pop() {

		extern __shared__ char shared[];
		Segment**		next 	= (Segment**)	&shared[0];
		int* 	 		pos 	= (int*) 		&shared[threadsPerBlock*sizeof(Segment*)];
		int* 	 		count 	= (int*) 		&shared[threadsPerBlock*(sizeof(Segment*)+sizeof(int))];
		int* 	 		size 	= (int*) 		&shared[threadsPerBlock*(sizeof(Segment*)+2*sizeof(int))];
		T* 		 		values	= (T*) 			&shared[threadsPerBlock*(sizeof(Segment*)+3*sizeof(int))];

		const int id = threadIdx.x;

		if (count[id] == 0) {

			//	printf("thread %i: pos=%i swap in\n", id, pos[id]);

			// swap in next segment

			if (pos[id] == 0) {

				//	printf("thread %i: delete segment\n", id);

				Segment* s = next[id];
				next[id] = next[id]->next;
				delete s;
				pos[id] = M;
			}

			for (int i = 0; i < N; i++) {
				values[index(i, id)] = next[id]->data[(pos[id] - 1) * N + i];
			}

			pos[id]--;
			count[id] += N;
		}

		T res = values[index(count[id] - 1, id)];
		count[id]--;
		size[id]--;

		return res;
	}

	__device__
	static void print() {

		extern __shared__ char shared[];
		Segment** 		next 	= (Segment**)	&shared[0];
		//int* const 		pos 	= (int*) 		&shared[threadsPerBlock*sizeof(Segment*)];
		int* 	 		count 	= (int*) 		&shared[threadsPerBlock*(sizeof(Segment*)+sizeof(int))];
		int* 	 		size 	= (int*) 		&shared[threadsPerBlock*(sizeof(Segment*)+2*sizeof(int))];
		T* 		 		values	= (T*) 			&shared[threadsPerBlock*(sizeof(Segment*)+3*sizeof(int))];

		const int id = threadIdx.x;

		if (id == 0) {

			printf("N=%i\n", N);

			int maxCount = 0;
			for (int j = 0; j < threadsPerBlock; j++) {
				if (count[j] > maxCount)
					maxCount = count[j];
			}

			for (int j = 0; j < threadsPerBlock; j++)
				printf("%3i ", size[j]);
			printf("\n");
			for (int j = 0; j < threadsPerBlock; j++)
				printf("----");
			printf("\n");

			for (int i = 0; i < 2 * N && i < maxCount; i++) {
				for (int j = 0; j < threadsPerBlock; j++) {
					if (i < count[j]) {
						printf("%3i ", values[index(i, j)]);
					} else {
						printf("    ");
					}
				}
				printf("\n");
			}
			for (int j = 0; j < threadsPerBlock; j++) {
				if (next[j] != nullptr) {
					printf("  * ");
				} else {
					printf("    ");
				}
			}
			printf("\n");
		}
		//__syncthreads();
	}

};

#endif /* MYSTACK_CUH_ */
