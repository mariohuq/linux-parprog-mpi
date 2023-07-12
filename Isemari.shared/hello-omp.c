#include <omp.h> 
#include <stdio.h> 
#include <stdlib.h> 

int main (int argc, char* argv[]) 
{ 
	int nthreads, tid; 
	// Начало параллельной области 
	#pragma omp parallel private(nthreads, tid)
	{ 
		// Получение номера потока
		tid = omp_get_thread_num(); 
		printf("Welcome to GFG from thread = %d\n", tid); 
		if (tid == 0) {
			// Выполняется только главным потоком
			nthreads = omp_get_num_threads();
			printf("Number of threads = %d\n", nthreads); 
		} 
	}
}