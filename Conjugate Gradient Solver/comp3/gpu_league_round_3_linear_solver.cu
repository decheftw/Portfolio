//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 3: sparse linear solver
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
// #include <thrust/raw_pointer_cast.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <vector>
#include <chrono>


using namespace std;

#define BDIM 16
#define BDIMY 16
#define BDIMX 8
//////////////////////////////////////////////////////////////////////////
////TODO 0: Please replace the following strings with your team name and author names
////Note: Please do not use space in the string, use "_" instead
//////////////////////////////////////////////////////////////////////////

namespace name
{
	std::string team = "dns";
	std::string author_1 = "Gao_Chen";
	std::string author_2 = "Nicolas_Flores";
	std::string author_3 = "Shikhar_Sinha";
};

//////////////////////////////////////////////////////////////////////////
////TODO: Read the following three CPU implementations for Jacobi, Gauss-Seidel, and Red-Black Gauss-Seidel carefully
////and understand the steps for these numerical algorithms
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
////These are the global variables that define the domain of the problem to solver
////You will need to use these parameters or macros in your GPU implementations
//////////////////////////////////////////////////////////////////////////
const int n = 128;							////grid size, we will change this value to up to 256 to test your code
const int g = 1;							////padding size
const int s = (n + 2 * g) * (n + 2 * g);			////array size
#define I(i,j) (i+g)*(n+2*g)+(j+g)		////2D coordinate -> array index
#define B(i,j) i<0||i>=n||j<0||j>=n		////check boundary
const bool verbose = false;				////set false to turn off print for x and residual
const double tolerance = 1e-3;			////tolerance for the iterative solver

//////////////////////////////////////////////////////////////////////////
////The following are three sample implementations for CPU iterative solvers
void Jacobi_Solver(double* x, const double* b)
{
	double* buf = new double[s];
	memcpy(buf, x, sizeof(double) * s);
	double* xr = x;			////read buffer pointer
	double* xw = buf;			////write buffer pointer
	int iter_num = 0;			////iteration number
	int max_num = 1e5;		////max iteration number
	double residual = 0.0;	////residual

	do {
		////update x values using the Jacobi iterative scheme
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				xw[I(i, j)] = (b[I(i, j)] + xr[I(i - 1, j)] + xr[I(i + 1, j)] + xr[I(i, j - 1)] + xr[I(i, j + 1)]) / 4.0;
			}
		}

		////calculate residual
		residual = 0.0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				residual += pow(4.0 * xw[I(i, j)] - xw[I(i - 1, j)] - xw[I(i + 1, j)] - xw[I(i, j - 1)] - xw[I(i, j + 1)] - b[I(i, j)], 2);
			}
		}

		 if (verbose)cout << "res: " << residual << endl;

		////swap the buffers
		double* swap = xr;
		xr = xw;
		xw = swap;
		iter_num++;
	} while (residual > tolerance && iter_num < max_num);

	x = xr;

	cout << "Jacobi solver converges in " << iter_num << " iterations, with residual " << residual << endl;

	delete[] buf;
}

void Gauss_Seidel_Solver(double* x, const double* b)
{
	int iter_num = 0;			////iteration number
	int max_num = 1e5;		////max iteration number
	double residual = 0.0;	////residual

	do {
		////update x values using the Gauss-Seidel iterative scheme
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				x[I(i, j)] = (b[I(i, j)] + x[I(i - 1, j)] + x[I(i + 1, j)] + x[I(i, j - 1)] + x[I(i, j + 1)]) / 4.0;
			}
		}

		////calculate residual
		residual = 0.0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				residual += pow(4.0 * x[I(i, j)] - x[I(i - 1, j)] - x[I(i + 1, j)] - x[I(i, j - 1)] - x[I(i, j + 1)] - b[I(i, j)], 2);
			}
		}

		// if (verbose)cout << "res: " << residual << endl;
		iter_num++;
	} while (residual > tolerance && iter_num < max_num);

	cout << "Gauss-Seidel solver converges in " << iter_num << " iterations, with residual " << residual << endl;
}

void Red_Black_Gauss_Seidel_Solver(double* x, const double* b)
{
	int iter_num=0;			////iteration number
	int max_num=1e5;		////max iteration number
	double residual=0.0;	////residual

	do{
		////red G-S
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				if((i+j)%2==0)		////Look at this line!
					x[I(i,j)]=(b[I(i,j)]+x[I(i-1,j)]+x[I(i+1,j)]+x[I(i,j-1)]+x[I(i,j+1)])/4.0;
			}
		}

		////black G-S
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				if((i+j)%2==1)		////And this line!
					x[I(i,j)]=(b[I(i,j)]+x[I(i-1,j)]+x[I(i+1,j)]+x[I(i,j-1)]+x[I(i,j+1)])/4.0;
			}
		}

		////calculate residual
		residual=0.0;
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				residual+=pow(4.0*x[I(i,j)]-x[I(i-1,j)]-x[I(i+1,j)]-x[I(i,j-1)]-x[I(i,j+1)]-b[I(i,j)],2);
			}
		}

		// if(verbose)cout<<"res: "<<residual<<endl;
		iter_num++;
	}while(residual>tolerance&&iter_num<max_num);	

	cout<<"Red-Black Gauss-Seidel solver converges in "<<iter_num<<" iterations, with residual "<<residual<<endl;

}

//////////////////////////////////////////////////////////////////////////
////In this function, we are solving a Poisson equation -laplace(p)=b, with p=x^2+y^2 and b=4.
////The boundary conditions are set on the one-ring ghost cells of the grid
//////////////////////////////////////////////////////////////////////////

void Test_CPU_Solvers()
{
	double* x = new double[s];
	memset(x, 0x0000, sizeof(double) * s);
	double* b = new double[s];
	for (int i = -1; i <= n; i++) {
		for (int j = -1; j <= n; j++) {
			b[I(i, j)] = 4.0;		////set the values for the right-hand side
		}
	}

	//////////////////////////////////////////////////////////////////////////
	////test Jacobi
	//for (int i = -1; i <= n; i++) {
	//	for (int j = -1; j <= n; j++) {
	//		if (B(i, j))
	//			x[I(i, j)] = (double)(i * i + j * j);	////set boundary condition for x
	//	}
	//}

	//Jacobi_Solver(x, b);

	//if (verbose) {
	//	cout << "\n\nx for Jacobi:\n";
	//	for (int i = 0; i < n; i++) {
	//		for (int j = 0; j < n; j++) {
	//			cout << x[I(i, j)] << ", ";
	//		}
	//		cout << std::endl;
	//	}
	//}


	cout << "\n\n";

	//////////////////////////////////////////////////////////////////////////
	////test Gauss-Seidel
	//memset(x, 0x0000, sizeof(double) * s);
	//for (int i = -1; i <= n; i++) {
	//	for (int j = -1; j <= n; j++) {
	//		if (B(i, j))
	//			x[I(i, j)] = (double)(i * i + j * j);	////set boundary condition for x
	//	}
	//}

	//Gauss_Seidel_Solver(x, b);

	//if (verbose) {
	//	cout << "\n\nx for Gauss-Seidel:\n";
	//	for (int i = 0; i < n; i++) {
	//		for (int j = 0; j < n; j++) {
	//			cout << x[I(i, j)] << ", ";
	//		}
	//		cout << std::endl;
	//	}
	//}
	//cout << "\n\n";

	//////////////////////////////////////////////////////////////////////////
	////test Red-Black Gauss-Seidel
	memset(x, 0x0000, sizeof(double) * s);
	for (int i = -1; i <= n; i++) {
		for (int j = -1; j <= n; j++) {
			if (B(i, j))
				x[I(i, j)] = (double)(i * i + j * j);	////set boundary condition for x
		}
	}
	auto cpu_start = chrono::system_clock::now();
	Red_Black_Gauss_Seidel_Solver(x, b);
	auto cpu_end = chrono::system_clock::now();
	chrono::duration<double> cpu_time = cpu_end - cpu_start;
	cout << "CPU runtime: " << cpu_time.count() * 1000. << " ms." << endl;

	if (verbose) {
		cout << "\n\nx for Red-Black Gauss-Seidel:\n";
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				cout << x[I(i, j)] << ", ";
			}
			cout << std::endl;
		}
	}
	cout << "\n\n";

	//////////////////////////////////////////////////////////////////////////

	delete[] x;
	delete[] b;
}

//////////////////////////////////////////////////////////////////////////
////TODO 1: your GPU variables and functions start here

__global__ void GPU_SOLVER(
	double* xr,
	double* xw,
	const double* b,
	double* resid)
{
	__shared__ double xs[BDIM + 2 * g][BDIM + 2 * g];
	__shared__ double bs[BDIM][BDIM];
	// first, load in the shared values, how to do this efficiently into xs
	// do it offset by whatever.
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	// NOTE: THIS WORKS ONLY WHEN BDIM > 2 * g. If padding/boundary region becomes too large,
	// need to update how we load in data
	xs[threadIdx.y][threadIdx.x] = xr[I(row - g, col - g)];

	if (threadIdx.x < 2 * g)
	{
		xs[threadIdx.y][threadIdx.x + BDIM] = xr[I(row - g, col + BDIM-g)];
	}

	if (threadIdx.y < 2 * g)
	{
		xs[threadIdx.y + BDIM][threadIdx.x] = xr[I(row + BDIM-g, col-g)];
	}

	if (threadIdx.x < 2 * g && threadIdx.y < 2 * g)
	{
		xs[threadIdx.y + BDIM][threadIdx.x + BDIM] = xr[I(row + BDIM-g, col + BDIM-g)];
	}


	// load in the b's
	bs[threadIdx.y][threadIdx.x] = b[I(row, col)];

	__syncthreads();
	//NOTE: DOES IT MAKE SNESE TO DO RED BLACK... maybe, maybe not? seems like the writes back to xw will be messed up
	xw[I(row, col)] = (bs[threadIdx.y][threadIdx.x]
		+ xs[threadIdx.y - 1 + g][threadIdx.x + g]
		+ xs[threadIdx.y + 1 + g][threadIdx.x + g]
		+ xs[threadIdx.y + g][threadIdx.x - 1 + g]
		+ xs[threadIdx.y + g][threadIdx.x + 1 + g]) * 0.25;

	xs[threadIdx.y][threadIdx.x] = xw[I(row - g, col - g)];
	bs[threadIdx.y][threadIdx.x] = b[I(row, col)];

	// get in new updated values on boundaries
	__syncthreads();
	if (threadIdx.x < 2 * g)
	{
		xs[threadIdx.y][threadIdx.x + BDIM] = xw[I(row-g, col + BDIM-g)];
	}

	if (threadIdx.y < 2 * g)
	{
		xs[threadIdx.y + BDIM][threadIdx.x] = xw[I(row + BDIM-g, col-g)];
	}

	if (threadIdx.x < 2 * g && threadIdx.y < 2 * g)
	{
		xs[threadIdx.y + BDIM][threadIdx.x + BDIM] = xw[I(row + BDIM-g, col + BDIM-g)];
	}

	__syncthreads();

	double resid_not_squared = (4.0 * xs[threadIdx.y + g][threadIdx.x + g]
		- xs[threadIdx.y - 1 + g][threadIdx.x + g]
		- xs[threadIdx.y + 1 + g][threadIdx.x + g]
		- xs[threadIdx.y + g][threadIdx.x - 1 + g]
		- xs[threadIdx.y + g][threadIdx.x + 1 + g]
		- bs[threadIdx.y][threadIdx.x]);
	resid[I(row, col)] = resid_not_squared * resid_not_squared;

	// printf("Resid: %f\n", resid[I(row,col)]);

}

__global__ void UPDATE_VALUES(
	double* xr,
	double* xw,
	const double* b,
	double* resid)
{
	__shared__ double xs[BDIM + 2 * g][BDIM + 2 * g];
	//__shared__ double bs[BDIM][BDIM];

	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;


	xs[threadIdx.y][threadIdx.x] = xr[I(row - g, col - g)];

	if (threadIdx.x < 2 * g)
	{
		xs[threadIdx.y][threadIdx.x + BDIM] = xr[I(row - g, col + BDIM - g)];
	}

	if (threadIdx.y < 2 * g)
	{
		xs[threadIdx.y + BDIM][threadIdx.x] = xr[I(row + BDIM - g, col - g)];
	}

	if (threadIdx.x < 2 * g && threadIdx.y < 2 * g)
	{
		xs[threadIdx.y + BDIM][threadIdx.x + BDIM] = xr[I(row + BDIM - g, col + BDIM - g)];
	}


	// load in the b's
	/*bs[threadIdx.y][threadIdx.x] = b[I(row, col)];*/

	__syncthreads();
	//NOTE: DOES IT MAKE SNESE TO DO RED BLACK... maybe, maybe not? seems like the writes back to xw will be messed up
	xw[I(row, col)] = (b[I(row,col)]
		+ xs[threadIdx.y - 1 + g][threadIdx.x + g]
		+ xs[threadIdx.y + 1 + g][threadIdx.x + g]
		+ xs[threadIdx.y + g][threadIdx.x - 1 + g]
		+ xs[threadIdx.y + g][threadIdx.x + 1 + g]) * 0.25;
}

__global__ void UPDATE_VALUES_RB(
	double* xr,
	double* xw,
	const double* b,
	double* resid)
{
	__shared__ double xs[BDIMY + 2 * g][(BDIMX * 2) + 2 * g];

	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col_l = (blockDim.x * blockIdx.x * 2) + threadIdx.x;
	int odd_offset = threadIdx.y & 0x1;
	int even_offset = 1 - odd_offset;

	xs[threadIdx.y][threadIdx.x] = xr[I(row - g, col_l - g)];
	xs[threadIdx.y][threadIdx.x + BDIMX] = xr[I(row - g, col_l - g + BDIMX)];

	if (threadIdx.x < 2 * g)
	{
		xs[threadIdx.y][threadIdx.x + (BDIMX * 2)] = xr[I(row - g, col_l - g + (BDIMX * 2))];
	}
	if (threadIdx.y < 2 * g)
	{
		xs[threadIdx.y + BDIMY][threadIdx.x] = xr[I(row - g + BDIMY, col_l - g)];
		xs[threadIdx.y + BDIMY][threadIdx.x + BDIMX] = xr[I(row - g + BDIMY, col_l - g + BDIMX)];
	}
	if (threadIdx.y < 2 * g && threadIdx.x < 2 * g)
	{
		xs[threadIdx.y + BDIMY][threadIdx.x + (BDIMX << 1)] = xr[I(row - g + BDIMY, col_l - g + (BDIMX << 1))];
	}


	__syncthreads();

	xs[threadIdx.y + g][2 * threadIdx.x + g + odd_offset] = (b[I(row,col_l + odd_offset)]
		+ xs[threadIdx.y - 1 + g][2 * threadIdx.x + g + odd_offset]
		+ xs[threadIdx.y + 1 + g][2 * threadIdx.x + g + odd_offset]
		+ xs[threadIdx.y + g][2 * threadIdx.x - 1 + g + odd_offset]
		+ xs[threadIdx.y + g][2 * threadIdx.x + 1 + g + odd_offset]) * 0.25;
	
	__syncthreads();

	xs[threadIdx.y + g][2 * threadIdx.x + g + even_offset] = (b[I(row, col_l + even_offset)]
		+ xs[threadIdx.y - 1 + g][2 * threadIdx.x + g + even_offset]
		+ xs[threadIdx.y + 1 + g][2 * threadIdx.x + g + even_offset]
		+ xs[threadIdx.y + g][2 * threadIdx.x - 1 + g + even_offset]
		+ xs[threadIdx.y + g][2 * threadIdx.x + 1 + g + even_offset]) * 0.25;

	__syncthreads();

	xw[I(row,col_l)] = xs[threadIdx.y + g][threadIdx.x + g];
	xw[I(row, col_l + BDIMX)] = xs[threadIdx.y + g][threadIdx.x + BDIMX + g];
}

__global__ void UPDATE_RESID(
	double* xr,
	double* xw,
	const double* b,
	double* resid)
{
	__shared__ double xs[BDIM + 2 * g][BDIM + 2 * g];
	//__shared__ double bs[BDIM][BDIM];
	// first, load in the shared values, how to do this efficiently into xs
	// do it offset by whatever.
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	xs[threadIdx.y][threadIdx.x] = xw[I(row - g, col - g)];
	//bs[threadIdx.y][threadIdx.x] = b[I(row, col)];

	// get in new updated values on boundaries

	if (threadIdx.x < 2 * g)
	{
		xs[threadIdx.y][threadIdx.x + BDIM] = xw[I(row - g, col + BDIM - g)];
	}

	if (threadIdx.y < 2 * g)
	{
		xs[threadIdx.y + BDIM][threadIdx.x] = xw[I(row + BDIM - g, col - g)];
	}

	if (threadIdx.x < 2 * g && threadIdx.y < 2 * g)
	{
		xs[threadIdx.y + BDIM][threadIdx.x + BDIM] = xw[I(row + BDIM - g, col + BDIM - g)];
	}

	__syncthreads();
	double resid_not_squared = (4.0 * xs[threadIdx.y + g][threadIdx.x + g]
		- xs[threadIdx.y - 1 + g][threadIdx.x + g]
		- xs[threadIdx.y + 1 + g][threadIdx.x + g]
		- xs[threadIdx.y + g][threadIdx.x - 1 + g]
		- xs[threadIdx.y + g][threadIdx.x + 1 + g]
		- b[I(row,col)]);
	resid[I(row, col)] = resid_not_squared * resid_not_squared;
}




__global__ void L2_NORM(double* x, float* res)
{
	 __shared__ double data[BDIM * BDIM];
	//  __shared__ double data[BDIMY * BDIMX];

	int tid = blockDim.x * threadIdx.y + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	// int col = (blockDim.x * blockIdx.x) * 2 + threadIdx.x;

	//int idx = ((row * gridDim.x * blockDim.x) + col) << 3;
	 

	data[tid] = x[I(row, col)];

	__syncthreads();

	for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
		if (tid < s) {
			data[tid] += data[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		atomicAdd(&res[0], data[0]);
	}



}




////Your implementations end here
//////////////////////////////////////////////////////////////////////////

ofstream out;

//////////////////////////////////////////////////////////////////////////
////GPU test function
void Test_GPU_Solver()
{
	double* x = new double[s];
	memset(x, 0x0000, sizeof(double) * s);
	double* b = new double[s];

	//////////////////////////////////////////////////////////////////////////
	////initialize x and b
	for (int i = -1; i <= n; i++) {
		for (int j = -1; j <= n; j++) {
			b[I(i, j)] = 4.0;		////set the values for the right-hand side
		}
	}
	for (int i = -1; i <= n; i++) {
		for (int j = -1; j <= n; j++) {
			if (B(i, j))
				x[I(i, j)] = (double)(i * i + j * j);	////set boundary condition for x
		}
	}

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time = 0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);


	//////////////////////////////////////////////////////////////////////////
	////TODO 2: call your GPU functions here
	////Requirement: You need to copy data from the CPU arrays, conduct computations on the GPU, and copy the values back from GPU to CPU
	////The final positions should be stored in the same place as the CPU function, i.e., the array of x
	////The correctness of your simulation will be evaluated by the residual (<1e-3)
	//////////////////////////////////////////////////////////////////////////
	double* xr_dev;
	cudaMalloc((void**)&xr_dev, s * sizeof(double));
	//cudaMemset(xr_dev, 0, s * sizeof(double));
	cudaMemcpy(xr_dev, x, s * sizeof(double), cudaMemcpyHostToDevice);

	double* xw_dev;
	cudaMalloc((void**)&xw_dev, s * sizeof(double));
	//cudaMemset(xw_dev, 0, s * sizeof(double));
	cudaMemcpy(xw_dev, x, s * sizeof(double), cudaMemcpyHostToDevice);

	double* b_dev;
	cudaMalloc((void**)&b_dev, s * sizeof(double));
	cudaMemcpy(b_dev, b, s * sizeof(double), cudaMemcpyHostToDevice);

	double* resid_dev;
	cudaMalloc((void**)&resid_dev, s * sizeof(double));
	cudaMemset(resid_dev, 0, s * sizeof(double));

	float* resid_sum;
	cudaMalloc((void**)&resid_sum,sizeof(float));
	cudaMemset(resid_sum, 0,sizeof(float));

	// int n_squared = n * n;
	// float resid_host = 1;
	int i = 0;
	double resid = 1;
	// while (resid_host > tolerance)
	while(resid > tolerance)
	{
		for (int j = 0; j < 100; j++)
		{
			if (j % 2 == 0)
			{
				//UPDATE_VALUES << <dim3(n / BDIM, n / BDIM), dim3(BDIM, BDIM) >> > (xr_dev, xw_dev, b_dev, resid_dev);
				UPDATE_VALUES_RB << <dim3(n /(2 * BDIMX), n / BDIMY), dim3(BDIMX, BDIMY) >> > (xr_dev, xw_dev, b_dev, resid_dev);

			}
			else
			{
				//UPDATE_VALUES << <dim3(n / BDIM, n / BDIM), dim3(BDIM, BDIM) >> > (xw_dev, xr_dev, b_dev, resid_dev);
				UPDATE_VALUES_RB << <dim3(n /(2 * BDIMX), n / BDIMY), dim3(BDIMX, BDIMY) >> > (xw_dev, xr_dev, b_dev, resid_dev);

			}
		}
		UPDATE_RESID << <dim3(n / BDIM, n / BDIM), dim3(BDIM, BDIM) >> > (xw_dev, xr_dev, b_dev, resid_dev);
		// L2_NORM << <dim3(n / (BDIM), n / BDIM), dim3(BDIM, BDIM) >> > (resid_dev, resid_sum);
		thrust::device_ptr<double> dev_ptr2(resid_dev);
		resid = thrust::reduce(dev_ptr2,dev_ptr2 + s,(double)0,thrust::plus<double>());
		// cout<<"Resid: "<<resid<<endl;
		// L2_NORM << <dim3(n / ( 2 * BDIMX), n / BDIMY), dim3(BDIMX, BDIMY) >> > (resid_dev, resid_sum);

		// cudaMemcpy(&resid_host, resid_sum, sizeof(float), cudaMemcpyDeviceToHost);
		// cudaMemset(resid_sum, 0, sizeof(float));
		i += 100;
	}

	if (i % 2 == 0)
	{
		cudaMemcpy(x, xr_dev, s * sizeof(double), cudaMemcpyDeviceToHost);
	}
	else
	{
		cudaMemcpy(x, xw_dev, s * sizeof(double), cudaMemcpyDeviceToHost);
	}
	
	cudaFree(xr_dev);
	cudaFree(xw_dev);
	cudaFree(b_dev);
	cudaFree(resid_dev);
	cudaFree(resid_sum);



	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, start, end);
	printf("\nGPU runtime: %.4f ms\n", gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	//////////////////////////////////////////////////////////////////////////

	////output x
	if (verbose) {
		cout << "\n\nx for your GPU solver:\n";
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				cout << x[I(i, j)] << ", ";
			}
			cout << std::endl;
		}
	}

	////calculate residual
	double residual = 0.0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			residual += pow(4.0 * x[I(i, j)] - x[I(i - 1, j)] - x[I(i + 1, j)] - x[I(i, j - 1)] - x[I(i, j + 1)] - b[I(i, j)], 2);
		}
	}
	cout << "\n\nresidual for your GPU solver: " << residual << endl;
	cout << "Iterations: " << i << endl;

	out << "R0: " << residual << endl;
	out << "T1: " << gpu_time << endl;

	//////////////////////////////////////////////////////////////////////////

	delete[] x;
	delete[] b;
}

int main()
{
	if (name::team == "Team_X") {
		printf("\nPlease specify your team name and team member names in name::team and name::author to start.\n");
		return 0;
	}

	std::string file_name = name::team + "_competition_3_linear_solver.dat";
	out.open(file_name.c_str());

	if (out.fail()) {
		printf("\ncannot open file %s to record results\n", file_name.c_str());
		return 0;
	}

	// Test_CPU_Solvers();	////You may comment out this line to run your GPU solver only
	Test_GPU_Solver();	////Test function for your own GPU implementation

	return 0;
}
