//////////////////////////////////////////////////////////////////////////
////This is the code implementation for Hanon finger exercise -- threads
////Dartmouth COSC89.25/189.03, GPU Programming and High-Performance Computing
////Team dns: Shikhar Sinha Gao Chen Nicolas Flores
//////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <vector>
#include <fstream>
using namespace std;

//////////////////////////////////////////////////////////////////////////
////TODO 0: Please replace the following strings with your team name and author names
////Note: Please do not use space in the string, use "_" instead
//////////////////////////////////////////////////////////////////////////

namespace name
{
	std::string team="dns";
	std::string author_1="Gao_Chen";
	std::string author_2="Nicolas_Flores";
	std::string author_3="Shikhar_Sinha";	////optional
};

//////////////////////////////////////////////////////////////////////////
////Hanon finger exercise for threads, Section 1, 1D array with 1D threads
////Learn how to use the built-in variables threadIdx, blockIdx, gridDim, and blockDim

////Exercise 1 is a sample kernel function, you don't need to implement this function
////Kernel dimension: <<<1,64>>>
////Expected output: 0,1,2,3,4,...,61,62,63
__global__ void Hanon_Exercise_0(int* array)
{
	array[threadIdx.x]=threadIdx.x;
}

////Kernel dimension: <<<1,64>>>
////Expected array values: 0,1,0,1,...,0,1
__global__ void Hanon_Exercise_1(int* array)
{
	/*TODO: Your implementation*/
	int threadID = threadIdx.x;
	array[threadID] = threadID % 2;
}

////Kernel dimension: <<<1,64>>>
////Expected array values: 1,2,3,4,1,2,3,4,...,1,2,3,4
__global__ void Hanon_Exercise_2(int* array)
{
	/*TODO: Your implementation*/
	int threadID = threadIdx.x;
	array[threadID] = (threadID % 4) + 1;
}

////Kernel dimension: <<<1,64>>>
////Expected array values: 0,1,2,3,4,3,2,1,0,1,2,3,4,3,2,1,...,0,1,2,3,4,3,2,1
__global__ void Hanon_Exercise_3(int* array)
{
	/*TODO: Your implementation*/
	int threadID = threadIdx.x;
	int modVal = threadID % 8;
	array[threadID] = (modVal < 5 ? modVal : 8 - modVal);
}

//////////////////////////////////////////////////////////////////////////
////Hanon finger exercise for threads, Section 2, 1D array with specified-dimension threads

////Kernel dimension: <<<dim3(2,2,1),dim3(4,4,1)>>>
////Expected output: 0,1,2,3,4,...,61,62,63
__global__ void Hanon_Exercise_4(int* array)
{
	/*TODO: Your implementation*/
	int blockID = blockIdx.x + blockIdx.y * gridDim.x;
	int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	array[threadID] = threadID;
}

////Kernel dimension: <<<dim3(2,2,1),dim3(4,4,1)>>>
////Expected array values: 0,1,0,1,...,0,1
__global__ void Hanon_Exercise_5(int* array)
{
	/*TODO: Your implementation*/
	int blockID = blockIdx.x + blockIdx.y * gridDim.x;
	int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	array[threadID] = threadID % 2;
}

////Kernel dimension: <<<8,dim3(2,4)>>>
////Expected array values: 1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,...,16,16,16,16
__global__ void Hanon_Exercise_6(int* array)
{
	/*TODO: Your implementation*/
	int threadID = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	array[threadID] = (threadID / 4) + 1;
}

////Kernel dimension: <<<8,dim3(2,2,2)>>>
////Expected array values: 1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,...,1,2,3,4,5,6,7,8
__global__ void Hanon_Exercise_7(int* array)
{
	int threadID = blockIdx.x * blockDim.x * blockDim.y * blockDim.z 
					+ threadIdx.z * blockDim.y * blockDim.x
					+ threadIdx.y * blockDim.x + threadIdx.x;
	array[threadID] = (threadID % 8) + 1;
	/*TODO: Your implementation*/
}

//////////////////////////////////////////////////////////////////////////
////Hanon finger exercise for threads, Section 3, 2D array

////Here we declare a 2D array on device with the size of 8x8
__device__ int b_on_dev[8][8];

////Kernel dimension: <<<1,64>>>
////Expected 2D array values:
////0       1       2       3       4       5       6       7
////8       9       10      11      12      13      14      15
////16      17      18      19      20      21      22      23
////24      25      26      27      28      29      30      31
////32      33      34      35      36      37      38      39
////40      41      42      43      44      45      46      47
////48      49      50      51      52      53      54      55
////56      57      58      59      60      61      62      63
__global__ void Hanon_Exercise_8()
{
	b_on_dev[threadIdx.x / 8][threadIdx.x % 8] = threadIdx.x;
	/*TODO: Your implementation*/
	////Hint: assign values to b_on_dev, e.g., b_on_dev[threadIdx.x][threadIdx.y]=1
}

////Kernel dimension: <<<1,dim3(8,8)>>>
////Expected 2D array values: the same as Exercise 8
__global__ void Hanon_Exercise_9()
{
	int threadID = threadIdx.y * blockDim.x + threadIdx.x;
	b_on_dev[threadID / 8][threadID % 8] = threadID;


	/*TODO: Your implementation*/
}

////Kernel dimension: <<<1,dim3(8,8)>>>
////Expected 2D array values: the transpose of Exercise 8, i.e.:
////0       8       16      24      32      40      48      56
////1       9       17      25      33      41      49      57
////2       10      18      26      34      42      50      58
////3       11      19      27      35      43      51      59
////4       12      20      28      36      44      52      60
////5       13      21      29      37      45      53      61
////6       14      22      30      38      46      54      62
////7       15      23      31      39      47      55      63
__global__ void Hanon_Exercise_10()
{
	int threadID = threadIdx.y * blockDim.x + threadIdx.x;
	b_on_dev[threadID % 8][threadID / 8] = threadID;

	/*TODO: Your implementation*/
}

////Kernel dimension: <<<dim3(2,2),dim3(4,4)>>>
////Expected 2D array values: 4 repeated blocks
////0       1       2       3       0       1       2       3
////4       5       6       7       4       5       6       7
////8       9       10      11      8       9       10      11
////12      13      14      15      12      13      14      15
////0       1       2       3       0       1       2       3
////4       5       6       7       4       5       6       7
////8       9       10      11      8       9       10      11
////12      13      14      15      12      13      14      15
__global__ void Hanon_Exercise_11()
{
	/*TODO: Your implementation*/
	b_on_dev[(blockIdx.y * blockDim.y) + threadIdx.y][ (blockIdx.x * blockDim.x) + threadIdx.x] = threadIdx.y * blockDim.x + threadIdx.x;
}


////Your tasks are all done here!
//////////////////////////////////////////////////////////////////////////


ofstream out;

////Helper function: copy the device array to host and print
__host__ void Print_Array_On_Device(const int test_id,const int* array_on_device,const int s)
{
	std::vector<int> array_on_host(s);	
	cudaMemcpy(&array_on_host[0],array_on_device,s*sizeof(int),cudaMemcpyDeviceToHost);

	printf("\nHanon exercise %d:\n",test_id);
	out<<"\nHanon exercise "<<test_id<<endl;
	for(int i=0;i<s;i++)printf("%d ",array_on_host[i]);printf("\n");
	for(int i=0;i<s;i++)out<<array_on_host[i]<<" ";out<<endl;
}

////Helper function: copy the device array to host and print
__host__ void Print_b_On_Device(const int test_id)
{
	int b_on_host[8][8];
	cudaMemcpyFromSymbol((void*)b_on_host,b_on_dev,64*sizeof(int));

	printf("\nHanon exercise %d:\n",test_id);
	out<<"\nHanon exercise "<<test_id<<endl;
	for(int i=0;i<8;i++){
		for(int j=0;j<8;j++){
			printf("%d\t",b_on_host[i][j]);
			out<<b_on_host[i][j]<<"\t";
		}
		printf("\n");
		out<<endl;
	}
	printf("\n");
	out<<endl;
}

////Test your implementation for exercises
////Note: Please do not change this function!
__host__ void Hanon_Exercise_Test()
{
	////allocate array on device
	const int s=64;
	int* array_on_device=0;
	cudaMalloc((void**)&array_on_device,s*sizeof(int));

	cudaMemset(array_on_device,0,s*sizeof(int));
	Hanon_Exercise_0<<<1,64>>>(array_on_device);
	Print_Array_On_Device(0,array_on_device,s);

	cudaMemset(array_on_device,0,s*sizeof(int));
	Hanon_Exercise_1<<<1,64>>>(array_on_device);
	Print_Array_On_Device(1,array_on_device,s);

	cudaMemset(array_on_device,0,s*sizeof(int));
	Hanon_Exercise_2<<<1,64>>>(array_on_device);
	Print_Array_On_Device(2,array_on_device,s);

	cudaMemset(array_on_device,0,s*sizeof(int));   
	Hanon_Exercise_3<<<1,64>>>(array_on_device);
	Print_Array_On_Device(3,array_on_device,s);

	cudaMemset(array_on_device,0,s*sizeof(int));
	Hanon_Exercise_4<<<dim3(2,2,1),dim3(4,4,1)>>>(array_on_device);
	Print_Array_On_Device(4,array_on_device,s);

	cudaMemset(array_on_device,0,s*sizeof(int));
	Hanon_Exercise_5<<<dim3(2,2,1),dim3(4,4,1)>>>(array_on_device);
	Print_Array_On_Device(5,array_on_device,s);

	cudaMemset(array_on_device,0,s*sizeof(int));
	Hanon_Exercise_6<<<8,dim3(2,4)>>>(array_on_device);
	Print_Array_On_Device(6,array_on_device,s);

	cudaMemset(array_on_device,0,s*sizeof(int));
	Hanon_Exercise_7<<<8,dim3(2,2,2)>>>(array_on_device);
	Print_Array_On_Device(7,array_on_device,s);

	int* b_on_dev_ptr=0;
	cudaGetSymbolAddress((void**)&b_on_dev_ptr,b_on_dev);
	cudaMemset(b_on_dev_ptr,0,64*sizeof(int));
	Hanon_Exercise_8<<<1,64>>>();
	printf("\nHanon exercise 8:\n");
	Print_b_On_Device(8);

	b_on_dev_ptr=0;
	cudaGetSymbolAddress((void**)&b_on_dev_ptr,b_on_dev);
	cudaMemset(b_on_dev_ptr,0,64*sizeof(int));
	Hanon_Exercise_9<<<1,dim3(8,8)>>>();
	printf("\nHanon exercise 9:\n");
	Print_b_On_Device(9);

	b_on_dev_ptr=0;
	cudaGetSymbolAddress((void**)&b_on_dev_ptr,b_on_dev);
	cudaMemset(b_on_dev_ptr,0,64*sizeof(int));
	Hanon_Exercise_10<<<1,dim3(8,8)>>>();
	printf("\nHanon exercise 10:\n");
	Print_b_On_Device(10);

	b_on_dev_ptr=0;
	cudaGetSymbolAddress((void**)&b_on_dev_ptr,b_on_dev);
	cudaMemset(b_on_dev_ptr,0,64*sizeof(int));
	Hanon_Exercise_11<<<dim3(2,2),dim3(4,4)>>>();
	printf("\nHanon exercise 11:\n");
	Print_b_On_Device(11);
}

int main()
{
	if(name::team=="TeamX"){
		printf("\nPlease specify your team name and team member names in name::team and name::author to start.\n");
		return 0;
	}

	std::string file_name=name::team+"_exercise_thread.dat";
	out.open(file_name.c_str());
	
	if(out.fail()){
		printf("\ncannot open file %s to record results\n",file_name.c_str());
		return 0;
	}

	Hanon_Exercise_Test();
	return 0;
}
