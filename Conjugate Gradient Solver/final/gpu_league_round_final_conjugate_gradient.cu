//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round Final: conjugate gradient solver
//////////////////////////////////////////////////////////////////////////
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
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
#include <omp.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
using namespace std;

#define PRECONDITION 0
#define BDIMX 64

//////////////////////////////////////////////////////////////////////////
////TODO 0: Please replace the following strings with your team name and author names
////Note: Please do not use space in the string, use "_" instead
//////////////////////////////////////////////////////////////////////////

namespace name
{
	std::string team="dns";
	std::string author_1="Gao_Chen";
	std::string author_2="Nicolas_Flores";
	std::string author_3="Shikhar_Sinha";
};

//////////////////////////////////////////////////////////////////////////
////This project implements the conjugate gradient solver to solve sparse linear systems
////For the mathematics, please read https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
////The algorithm we are implementing is in Page 50, Algorithm B.2, the standard conjugate gradient (without a preconditioner)
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
////These are the global variables that define the domain of the problem to solver (for both CPU and GPU)

const int grid_size = 256;										////grid size, we will change this value to up to 256 to test your code, notice that we do not have padding elementsd
const int s=grid_size*grid_size;								////array size
#define I(i,j) ((i)*grid_size+(j))								////2D coordinate -> array index
#define B(i,j) (i)<0||(i)>=grid_size||(j)<0||(j)>=grid_size		////check boundary
const bool verbose=false;										////set false to turn off print for x and residual
const int max_iter_num=10000;									////max cg iteration number
const double tolerance=1e-3;									////tolerance for the iterative solver

//////////////////////////////////////////////////////////////////////////
////TODO 1: Warm up practice 1 -- implement a function for (sparse)matrix-(dense)vector multiplication and a function for vector-vector dot product

////calculate mv=M*v, here M is a square matrix
void MV(/*CRS sparse matrix*/const double* val,const int* col,const int* ptr,/*number of column*/const int n,/*input vector*/const double* v,/*result*/double* mv)
{
    /*Your implementation starts*/
	#pragma omp parallel for
	for(int i = 0; i < n; i++)
	{
		mv[i] = 0.0;
		int idx = ptr[i];
		int end = ptr[i+1];
		while (idx < end)
		{
			int v_pos = col[idx];
			mv[i] += val[idx] * v[v_pos];
			idx++;
		}
	}
	/*Your implementation ends*/
}

////return the dot product between a and b
double Dot(const double* a,const double* b,const int n)
{
    /*Your implementation starts*/
	double product = 0.0;
	#pragma omp parallel for reduction (+:product)
    for(int i = 0; i < n; i++)
    {
			product += a[i] * b[i];
	}
	#pragma omp barrier
	/*Your implementation ends*/
	return product;
}

//////////////////////////////////////////////////////////////////////////
////TODO 2: Warm up practice 2 -- implement a CPU-based conjugate gradient solver based on the painless PCG course notes to solve Ax=b
////Please read the notes and implement all the TODOs in the function

void Conjugate_Gradient_Solver(const double* val,const int* col,const int* ptr,const int n,		////A is an n x n sparse matrix stored in CRS format
								double* r,double* q,double* d,									////intermediate variables
								double* x,const double* b,										////x and b
								const int max_iter,const double tol)							////solver parameters
{
	////declare variables
	int iter=0;
	double delta_old=0.0;
	double delta_new=0.0;
	double alpha=0.0;
	double beta=0.0;
	////TODO: r=b-Ax


    MV(val, col, ptr, n, x, r);
    // At this pt, r = Ax

	#pragma omp parallel for
    for(int i = 0; i < n; i++)
    {
        r[i] = b[i] - r[i];
    }


    ////TODO: d=r
    
	memcpy(&d[0], &r[0], n * sizeof(double));
	
	#pragma omp parallel for
	for(int i = 0; i < n; i++)
	{
		d[i] = r[i];
	}
	

	
    ////TODO: delta_new=rTr
    
    delta_new = Dot(r, r, n);
	
	////Here we use the absolute tolerance instead of a relative one, which is slightly different from the notes
	while(iter<max_iter&& delta_new>tol){	
        ////TODO: q=Ad
        MV(val, col, ptr, n, d, q);

        ////TODO: alpha=delta_new/d^Tq
        alpha = delta_new / Dot(d, q, n);

		////TODO: x=x+alpha*d
		#pragma omp parallel for
        for(int i = 0; i < n; i++)
        {
            x[i] += alpha * d[i];
        }


		if(iter%50==0&&iter>1){
            ////TODO: r=b-Ax
            MV(val, col, ptr, n, x, r);
			// At this pt, r = Ax
			#pragma omp parallel for
            for(int i = 0; i < n; i++)
            {
                r[i] = b[i] - r[i];
            }
		}
		else{
			////TODO: r=r-alpha*q
			#pragma omp parallel for
            for (int i = 0; i < n; i++)
            {
                r[i] -= alpha * q[i];
            }
		}

        ////TODO: delta_old=delta_new
        
        delta_old = delta_new;

        ////TODO: delta_new=r^Tr
        
        delta_new = Dot(r,r,n);

        ////TODO: beta=delta_new/delta_old
        
        beta =  delta_new / delta_old;
		
        ////TODO: d=r+beta*d
    	#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            d[i] = r[i] + beta * d[i];
        }

        ////TODO: increase the counter
        iter++;
	}

	if(iter<max_iter)
		cout<<"CPU conjugate gradient solver converges after "<<iter<<" iterations with residual "<<(delta_new)<<endl;
	else 
		cout<<"CPU conjugate gradient solver does not converge after "<<max_iter<<" iterations with residual "<<(delta_new)<<endl;
}

//////////////////////////////////////////////////////////////////////////
////TODO 3: implement your GPU-based conjugate gradient solver
////Put your CUDA variables and functions here


__global__ void MV_GPU(
						/*CRS sparse matrix*/const double* val, const int* col, const int* ptr,
						/*number of column*/const int* n,
						/*input vector*/const double* v,
						/*result*/double* mv)
{

	// Could do some cleverness here where the threads of the block work together to load in A
	// Could maybe use the fact that that A[I(i,i)] is nonzero to do somoe loads of the v vector
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int row_pos = ptr[row];
	int row_end = ptr[row + 1];
	double running_sum = 0.0;

	for (; row_pos < row_end; row_pos++)
	{
		int v_pos = col[row_pos];
		running_sum += val[row_pos] * v[v_pos];
	}
	mv[row] = running_sum;

}







struct my_func_add
{
	double a=0.0;
	my_func_add(double _a):a(_a){}

	__host__ __device__ double operator()(const double& x,const double& y) const
	{
		return x+a*y;
	}
};

struct my_func_subtract
{
	double a=0.0;
	my_func_subtract(double _a):a(_a){}

	__host__ __device__ double operator()(const double& x,const double& y) const
	{
		return x-a*y;
	}
};
//////////////////////////////////////////////////////////////////////////




ofstream out;

//////////////////////////////////////////////////////////////////////////
////Test functions
////Here we setup a test example by initializing the same Poisson problem as in the last competition: -laplace(p)=b, with p=x^2+y^2 and b=-4.
////The boundary conditions are set on the one-ring ghost cells of the grid
////There is nothing you need to implement in this function

void Initialize_2D_Poisson_Problem(vector<double>& val,vector<int>& col,vector<int>& ptr,vector<double>& b)
{
	////assemble the CRS sparse matrix
	////The grid dimension is grid_size x grid_size. 
	////The matrix's dimension is s x s, with s= grid_size*grid_size.
	////We also initialize the right-hand vector b

	val.clear();
	col.clear();
	ptr.resize(s+1,0);
	b.resize(s,-4.);

	for(int i=0;i<grid_size;i++){
		for(int j=0;j<grid_size;j++){
			int r=I(i,j);
			int nnz_for_row_r=0;

			////set (i,j-1)
			if(!(B(i,j-1))){
				int c=I(i,j-1);
				val.push_back(-1.);
				col.push_back(c);
				nnz_for_row_r++;
			}
			else{
				double boundary_val=(double)(i*i+(j-1)*(j-1));	
				b[r]+=boundary_val;
			}

			////set (i-1,j)
			if(!(B(i-1,j))){
				int c=I(i-1,j);
				val.push_back(-1.);
				col.push_back(c);
				nnz_for_row_r++;
			}
			else{
				double boundary_val=(double)((i-1)*(i-1)+j*j);
				b[r]+=boundary_val;
			}

			////set (i+1,j)
			if(!(B(i+1,j))){
				int c=I(i+1,j);
				val.push_back(-1.);
				col.push_back(c);
				nnz_for_row_r++;
			}
			else{
				double boundary_val=(double)((i+1)*(i+1)+j*j);
				b[r]+=boundary_val;
			}

			////set (i,j+1)
			if(!(B(i,j+1))){
				int c=I(i,j+1);
				val.push_back(-1.);
				col.push_back(c);
				nnz_for_row_r++;
			}
			else{
				double boundary_val=(double)(i*i+(j+1)*(j+1));
				b[r]+=boundary_val;
			}

			////set (i,j)
			{
				val.push_back(4.);
				col.push_back(r);
				nnz_for_row_r++;
			}
			ptr[r+1]=ptr[r]+nnz_for_row_r;
		}
	}
}

//////////////////////////////////////////////////////////////////////////
////CPU test function
////There is nothing you need to implement in this function
void Test_CPU_Solvers()
{
	vector<double> val;
	vector<int> col;
	vector<int> ptr;
	vector<double> b;
	Initialize_2D_Poisson_Problem(val,col,ptr,b);

	vector<double> x(s,0.);
	vector<double> r(s,0.);
	vector<double> q(s,0.);
	vector<double> d(s,0.);
	
	auto start=chrono::system_clock::now();

	Conjugate_Gradient_Solver(&val[0],&col[0],&ptr[0],s,
								&r[0],&q[0],&d[0],
								&x[0],&b[0],
								max_iter_num,tolerance);

	auto end=chrono::system_clock::now();
	chrono::duration<double> t=end-start;
	double cpu_time=t.count()*1000.;	

	if(verbose){
		cout<<"\n\nx for CG on CPU:\n";
		for(int i=0;i<s;i++){
			cout<<x[i]<<", ";
		}	
	}
	cout<<"\n\n";

	//////calculate residual
	MV(&val[0],&col[0],&ptr[0],s,&x[0],&r[0]);
	for(int i=0;i<s;i++)r[i]=b[i]-r[i];
	double residual=Dot(&r[0],&r[0],s);
	cout<<"\nCPU time: "<<cpu_time<<" ms"<<endl;
	cout<<"Residual for your CPU solver: "<<residual<<endl;

	out<<"R0: "<<residual<<endl;
	out<<"T0: "<<cpu_time<<endl;
}

//////////////////////////////////////////////////////////////////////////
////GPU test function
void Test_GPU_Solver()
{
	if (s < BDIMX) {
		cout << "BDIMX is " << BDIMX << " which is too low for grid size " << grid_size << endl;
		cout << "Set BDIMX to at most grid_size^2" << endl;
		return;
	}

	vector<double> val;
	vector<int> col;
	vector<int> ptr;
	vector<double> b;
	Initialize_2D_Poisson_Problem(val,col,ptr,b);

	vector<double> x(s,0.);
	vector<double> r(s,0.);
	vector<double> q(s,0.);
	vector<double> d(s,0.);
	
	// Precondtioner
	vector<double> a(s,0.); // This is a new vector used in the algorithm

	vector<double> m_val(s,0.);
	vector<int> m_col(s,0.);
	vector<int> m_ptr(s+1,0.);
	if (PRECONDITION) {
		// Initialize_Preconditioner(m_val, m_col, m_ptr, val, col, ptr);
		for (int i = 0; i < s; i++) {
			int offset = ptr[i];
			while (col[offset] != i) {
				offset += 1;
			}
			
			m_val[i] = 1 / val[offset];
			m_col[i] = i;
			m_ptr[i] = i;
		}
		// CRS format, attach the # of elements to the end of the ptr array
		m_ptr[s] = s;
	}

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//////////////////////////////////////////////////////////////////////////
	////TODO 4: call your GPU functions here
	////Requirement: You need to copy data from the CPU arrays, conduct computations on the GPU, and copy the values back from GPU to CPU
	////The final variables should be stored in the same place as the CPU function, i.e., the array of x
	////The correctness of your simulation will be evaluated by the residual (<1e-3)
	//////////////////////////////////////////////////////////////////////////

	double *val_dev;
	int *col_dev;
	int *ptr_dev;

	cudaMalloc((void**)&val_dev, val.size() * sizeof(double));
	cudaMemcpy(val_dev, &val[0], val.size() * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&col_dev, col.size() * sizeof(int));
	cudaMemcpy(col_dev, &col[0], col.size() * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&ptr_dev, ptr.size() * sizeof(int));
	cudaMemcpy(ptr_dev, &ptr[0], ptr.size() * sizeof(int), cudaMemcpyHostToDevice);

	double *m_val_dev;
	int *m_col_dev;
	int *m_ptr_dev;

	double *a_dev;

	// Copy a over
	cudaMalloc((void**)&a_dev, a.size() * sizeof(double));
	cudaMemcpy(a_dev, &a[0], a.size() * sizeof(double), cudaMemcpyHostToDevice);
	if (PRECONDITION) {
		// Copy m over
		cudaMalloc((void**)&m_val_dev, m_val.size() * sizeof(double));
		cudaMemcpy(m_val_dev, &m_val[0], m_val.size() * sizeof(double), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&m_col_dev, m_col.size() * sizeof(int));
		cudaMemcpy(m_col_dev, &m_col[0], m_col.size() * sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&m_ptr_dev, m_ptr.size() * sizeof(int));
		cudaMemcpy(m_ptr_dev, &m_ptr[0], m_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
		
	}

	double *x_dev;
	double *r_dev;
	double *q_dev;
	double *d_dev;
	double *b_dev;



	cudaMalloc((void**)&x_dev, x.size() * sizeof(double));
	cudaMemcpy(x_dev, &x[0], x.size() * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&r_dev, r.size() * sizeof(double));
	cudaMemcpy(r_dev, &r[0], r.size() * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&q_dev, q.size() * sizeof(double));
	cudaMemcpy(q_dev, &x[0], q.size() * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_dev, d.size() * sizeof(double));
	cudaMemcpy(d_dev, &d[0], d.size() * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&b_dev, b.size() * sizeof(double));
	cudaMemcpy(b_dev, &b[0], b.size() * sizeof(double), cudaMemcpyHostToDevice);

	thrust::device_ptr<double> m_thrust(m_val_dev);
	thrust::device_ptr<double> a_thrust(a_dev);
	thrust::device_ptr<double> x_thrust(x_dev);
	thrust::device_ptr<double> r_thrust(r_dev);
	thrust::device_ptr<double> q_thrust(q_dev);
	thrust::device_ptr<double> d_thrust(d_dev);
	thrust::device_ptr<double> b_thrust(b_dev);

	int *s_dev;
	cudaMalloc((void**)&s_dev, sizeof(int));
	cudaMemcpy(s_dev, &s, sizeof(int), cudaMemcpyHostToDevice);


	////declare variables
	int iter=0;
	double delta_old=0.0;
	double delta_new=0.0;
	double alpha=0.0;
	double beta=0.0;

	// r = b-Ax
	MV_GPU<<< s / BDIMX, BDIMX>>>(val_dev, col_dev, ptr_dev, s_dev, x_dev,  r_dev);

	// r = b-r
	thrust::transform(b_thrust, b_thrust + s, r_thrust, r_thrust, thrust::minus<double>());


	if (PRECONDITION) {
		// d = Mr
		thrust::transform(m_thrust, m_thrust+s, r_thrust, d_thrust, thrust::multiplies<double>());
		// MV_GPU<<< s / BDIMX, BDIMX>>>(m_val_dev, m_col_dev, m_ptr_dev, s_dev, r_dev,  d_dev);
	} else {
		// d=r
		cudaMemcpy(d_dev, r_dev, s * sizeof(double), cudaMemcpyDeviceToDevice);
	}


	if (PRECONDITION) {
		// delta_new = rTd
		delta_new = thrust::inner_product(r_thrust, r_thrust + s, d_thrust,(double)0.0);
	} else { 
		// delta_new=rTr
		delta_new = thrust::inner_product(r_thrust, r_thrust + s, r_thrust,(double)0.0);
	}
	
	// So that the Preconditioned CG converges to the same residual as non-Preconditioned CG
	double tolerance_ = PRECONDITION ? tolerance / 10 : tolerance;
	////Here we use the absolute tolerance instead of a relative one, which is slightly different from the notes
	while( iter < max_iter_num && delta_new > tolerance_){	
		// q=Ad
		MV_GPU<<< s / BDIMX, BDIMX>>>(val_dev, col_dev, ptr_dev, s_dev, d_dev,  q_dev);


		// alpha=delta_new/d^Tq
		double dT_q =  thrust::inner_product(d_thrust, d_thrust + s, q_thrust, (double)0.0);

		alpha = delta_new / dT_q;

		// x=x+alpha*d
		thrust::transform(/*src1*/x_thrust, x_thrust + s,/*src2*/d_thrust,/*des*/x_thrust, my_func_add(alpha));


		if(iter%50==0&&iter>1){
			// r=b-Ax
			MV_GPU<<< s / BDIMX, BDIMX>>>(val_dev, col_dev, ptr_dev, s_dev, x_dev,  r_dev);
			// r = b-r
			thrust::transform(b_thrust, b_thrust + s, r_thrust, r_thrust, thrust::minus<double>());
		}
		else{
			// r=r-alpha*q
			////customized transform, d2=d1+a*d2
			thrust::transform(/*src1*/r_thrust, r_thrust + s,/*src2*/q_thrust,/*des*/r_thrust, my_func_subtract(alpha));

		}
		
		// a = Mr (NOTE: this 'a' is called 's' in the quake-paper algorithm)
		if (PRECONDITION) {
			thrust::transform(m_thrust, m_thrust + s, r_thrust, a_thrust, thrust::multiplies<double>());
		}

		// delta_old=delta_new
		delta_old = delta_new;

		if (PRECONDITION) {
			// delta_new = r^Ta 
			delta_new = thrust::inner_product(r_thrust, r_thrust + s, a_thrust,(double)0.0);
		} else { 
			// delta_new=r^Tr
			delta_new = thrust::inner_product(r_thrust, r_thrust + s, r_thrust,(double)0.0);
		}


		// beta=delta_new/delta_old
		beta = delta_new / delta_old;
		
		////customized transform, d2=d1+a*d2
		if (PRECONDITION) {
			// d = a + beta*d
			thrust::transform(/*src1*/a_thrust, a_thrust + s,/*src2*/d_thrust,/*des*/d_thrust, my_func_add(beta));
		} else {
			// d=r+beta*d
			thrust::transform(/*src1*/r_thrust, r_thrust + s,/*src2*/d_thrust,/*des*/d_thrust, my_func_add(beta));

		}

		iter++;
	}


	cudaMemcpy(&x[0], x_dev, s *sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(x_dev);
	cudaFree(r_dev);
	cudaFree(q_dev);
	cudaFree(d_dev);
	cudaFree(b_dev);
	cudaFree(val_dev);
	cudaFree(col_dev);
	cudaFree(ptr_dev);



	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	//////////////////////////////////////////////////////////////////////////
	cout<<"iters: "<<iter<<endl;
	if(verbose){
		cout<<"\n\nx for CG on GPU:\n";
		for(int i=0;i<grid_size;i++){
			for (int j=0;j<grid_size;j++) {
				cout<<x[I(i,j)]<<", ";
			}
		}	
	}
	cout<<"\n\n";

	//////calculate residual
	MV(&val[0],&col[0],&ptr[0],s,&x[0],&r[0]);
	for(int i=0;i<s;i++)r[i]=b[i]-r[i];
	double residual=Dot(&r[0],&r[0],s);
	cout<<"\nGPU time: "<<gpu_time<<" ms"<<endl;
	cout<<"Residual for your GPU solver: "<<residual<<endl;

	out<<"R1: "<<residual<<endl;
	out<<"T1: "<<gpu_time<<endl;
}

int main()
{
	if(name::team=="Team_X"){
		printf("\nPlease specify your team name and team member names in name::team and name::author to start.\n");
		return 0;
	}

	std::string file_name=name::team+"_competition_final_conjugate_gradient.dat";
	out.open(file_name.c_str());
	
	if(out.fail()){
		printf("\ncannot open file %s to record results\n",file_name.c_str());
		return 0;
	}

	Test_CPU_Solvers();
	Test_GPU_Solver();

	return 0;
}
