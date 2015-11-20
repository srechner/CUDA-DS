namespace stack {

template<int N, int M>
struct segment {
	int data[N * M];
	segment* next;
};

/**
 *  shared memory is divided into four parts:
 *
 *  segment<N,M>** next = (segment<N,M>**) shared;
 *  int* pos       = (int*) &next[blockDim.x];
 *  int* count     = (int*) &pos[blockDim.x];
 *  int* values    = (int*) &count[blockDim.x];
 */
extern __shared__ long shared[];

__device__ __forceinline__ int index(int i, int id) {
	return i * blockDim.x + id;
}

// inits empty stack
template<int N, int M>
__device__ void init() {

	int id = blockDim.x * blockIdx.x + threadIdx.x;
	segment<N, M>** next = (segment<N, M>**) shared;
	int* pos = (int*) &next[blockDim.x];
	int* count = (int*) &pos[blockDim.x];
	int* values = (int*) &count[blockDim.x];

	/*if(id == 0) {
	 printf("next   = %ul\n", next);
	 printf("count  = %ul\n", count);
	 printf("values = %ul\n", values);
	 }
	 __syncthreads();*/

	count[id] = 0;
	pos[id] = M;
	next[id] = NULL;
}

// returns true, of stack is empty, false if not
template<int N, int M>
__device__ bool empty() {

	int id = blockDim.x * blockIdx.x + threadIdx.x;
	segment<N, M>** next = (segment<N, M>**) shared;
	int* pos = (int*) &next[blockDim.x];
	int* count = (int*) &pos[blockDim.x];
	int* values = (int*) &count[blockDim.x];

	return next[id] == NULL && count[id] == 0 && pos[id] == 0;
}

// pushes element x onto stack
template<int N, int M>
__device__ void push(int x) {

	int id = blockDim.x * blockIdx.x + threadIdx.x;
	segment<N, M>** next = (segment<N, M>**) shared;
	int* pos = (int*) &next[blockDim.x];
	int* count = (int*) &pos[blockDim.x];
	int* values = (int*) &count[blockDim.x];

	// swap out
	if (count[id] == 2 * N) {

		//printf("thread %i: pos=%i swap out\n", id, pos[id]);

		if (pos[id] == M) {

			//printf("thread %i create new segment\n", id);

			// create new segment
			segment<N, M>* s = new segment<N, M>();
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
}

// pops upper element from stack (-1 if empty)
template<int N, int M>
__device__ int pop() {

	if (empty<N, M>())
		return -1;

	int id = blockDim.x * blockIdx.x + threadIdx.x;
	segment<N, M>** next = (segment<N, M>**) shared;
	int* pos = (int*) &next[blockDim.x];
	int* count = (int*) &pos[blockDim.x];
	int* values = (int*) &count[blockDim.x];

	if (count[id] == 0) {

		//printf("thread %i: pos=%i swap in\n", id, pos[id]);

		// swap in next segment

		if (pos[id] == 0) {

			//printf("thread %i: delete segment\n", id);

			segment<N, M>* s = next[id];
			next[id] = next[id]->next;
			delete[] s;
			pos[id] = M;
		}

		for (int i = 0; i < N; i++) {
			values[index(i, id)] = next[id]->data[ (pos[id]-1)*N+i];
		}

		pos[id]--;
		count[id] += N;
	}

	int res = values[index(count[id] - 1, id)];
	count[id]--;

	return res;
}

template<int N, int M>
__device__ void print() {

	int id = blockDim.x * blockIdx.x + threadIdx.x;
	segment<N, M>** next = (segment<N, M>**) shared;
	int* count = (int*) &next[blockDim.x];
	int* values = (int*) &count[blockDim.x];

	if (id == 0) {

		for (int j = 0; j < blockDim.x; j++)
			printf("%3i ", count[j]);
		printf("\n------------\n");

		for (int i = 0; i < 2 * N; i++) {
			for (int j = 0; j < blockDim.x; j++) {
				printf("%3i ", values[index(i, j)]);
			}
			printf("\n");
		}
		printf("\n");
	}
	__syncthreads();
}

}
