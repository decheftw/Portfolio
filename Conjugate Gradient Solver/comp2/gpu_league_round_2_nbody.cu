//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 2: n-body simulation
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
#define BDIMX 32
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
////Here is a sample function implemented on CPU for n-body simulation.

__host__ void N_Body_Simulation_CPU_Poorman(double* pos_x,double* pos_y,double* pos_z,		////position array
											double* vel_x,double* vel_y,double* vel_z,		////velocity array
											double* acl_x,double* acl_y,double* acl_z,		////acceleration array
											const double* mass,								////mass array
											const int n,									////number of particles
											const double dt,								////timestep
											const double epsilon_squared)					////epsilon to avoid 0-denominator
{		
	////Step 1: set particle accelerations to be zero
	memset(acl_x,0x00,sizeof(double)*n);
	memset(acl_y,0x00,sizeof(double)*n);
	memset(acl_z,0x00,sizeof(double)*n);

	////Step 2: traverse all particle pairs and accumulate gravitational forces for each particle from pairwise interactions
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			////skip calculating force for itself
			if(i==j) continue;

			////r_ij=x_j-x_i
			double rx=pos_x[j]-pos_x[i];
			double ry=pos_y[j]-pos_y[i];
			double rz=pos_z[j]-pos_z[i];

			////a_ij=m_j*r_ij/(r+epsilon)^3, 
			////noticing that we ignore the gravitational coefficient (assuming G=1)
			double dis_squared=rx*rx+ry*ry+rz*rz;
			double one_over_dis_cube=1.0/pow(sqrt(dis_squared+epsilon_squared),3);
			double ax=mass[j]*rx*one_over_dis_cube;
			double ay=mass[j]*ry*one_over_dis_cube;
			double az=mass[j]*rz*one_over_dis_cube;

			////accumulate the force to the particle
			acl_x[i]+=ax;
			acl_y[i]+=ay;
			acl_z[i]+=az;
		}
	}

	////Step 3: explicit time integration to update the velocity and position of each particle
	for(int i=0;i<n;i++){
		////v_{t+1}=v_{t}+a_{t}*dt
		vel_x[i]+=acl_x[i]*dt;
		vel_y[i]+=acl_y[i]*dt;
		vel_z[i]+=acl_z[i]*dt;

		////x_{t+1}=x_{t}+v_{t}*dt
		pos_x[i]+=vel_x[i]*dt;
		pos_y[i]+=vel_y[i]*dt;
		pos_z[i]+=vel_z[i]*dt;
	}
}


//////////////////////////////////////////////////////////////////////////
////TODO 1: your GPU variables and functions start here
__global__ void N_BODY_GPU(
					double* pos_x,double* pos_y,double* pos_z,		////position array
					double* vel_x,double* vel_y,double* vel_z,		////velocity array
					double* acl_x,double* acl_y,double* acl_z,		////acceleration array

					double* pos_x_2,double* pos_y_2,double* pos_z_2,		////position array
					double* vel_x_2,double* vel_y_2,double* vel_z_2,		////velocity array
					double* acl_x_2,double* acl_y_2,double* acl_z_2,
					const double* mass,								////mass array
					const int* n,									////number of particles
					const double* dt,								////timestep
					const double* epsilon_squared)					////epsilon to avoid 0-denominator
{


	__shared__ double x[BDIMX];
	__shared__ double y[BDIMX];
	__shared__ double z[BDIMX];
	__shared__ double m[BDIMX];

	double new_acl_x = 0;
	double new_acl_y = 0;
	double new_acl_z = 0;
	double eps2 = epsilon_squared[0]; 
	int bnum = blockDim.x * blockIdx.x + threadIdx.x; // the particle index
	double own_pos_x = pos_x[bnum];
	double own_pos_y = pos_y[bnum];
	double own_pos_z = pos_z[bnum];


	// need to load in values n/p times so we do:
	for(int i = 0; i < n[0] / BDIMX; i++)
	{
		x[threadIdx.x] = pos_x[i * BDIMX + threadIdx.x];
		y[threadIdx.x] = pos_y[i * BDIMX + threadIdx.x];
		z[threadIdx.x] = pos_z[i * BDIMX + threadIdx.x];
		m[threadIdx.x] = mass[i * BDIMX + threadIdx.x]; 

		__syncthreads();

		// do the calculation: for each of the p bodies loaded in, calculate the stuff
		for(int j = 0; j < BDIMX; j+=1)
		{
			double r_x = x[j] - own_pos_x;
			double r_y = y[j] - own_pos_y;
			double r_z = z[j] - own_pos_z;
			double dist_sqr = r_x * r_x + r_y * r_y + r_z * r_z + eps2;
			double dist_sixth = dist_sqr * dist_sqr * dist_sqr;
			double m_over_dis_cube = m[j]/sqrt(dist_sixth);

			new_acl_x += r_x * m_over_dis_cube;
			new_acl_y += r_y * m_over_dis_cube;
			new_acl_z += r_z * m_over_dis_cube;

		}
		__syncthreads();
	}
	acl_x_2[bnum] = new_acl_x;
	acl_y_2[bnum] = new_acl_y;
	acl_z_2[bnum] = new_acl_z;
	double new_v_x = vel_x[bnum] + new_acl_x * dt[0];
	double new_v_y = vel_y[bnum] + new_acl_y * dt[0];
	double new_v_z = vel_z[bnum] + new_acl_z * dt[0];

	pos_x_2[bnum] = own_pos_x + new_v_x * dt[0];
	pos_y_2[bnum] = own_pos_y + new_v_y * dt[0];
	pos_z_2[bnum] = own_pos_z + new_v_z * dt[0];
	vel_x_2[bnum] = new_v_x;
	vel_y_2[bnum] = new_v_y;
	vel_z_2[bnum] = new_v_z;




}
	

////Your implementations end here
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
////Test function for n-body simulator
ofstream out;

//////////////////////////////////////////////////////////////////////////
////Please do not change the values below
const double dt=0.001;							////time step
const int time_step_num=10;						////number of time steps
const double epsilon=1e-2;						////epsilon added in the denominator to avoid 0-division when calculating the gravitational force
const double epsilon_squared=epsilon*epsilon;	////epsilon squared

////We use grid_size=4 to help you debug your code, change it to a bigger number (e.g., 16, 32, etc.) to test the performance of your GPU code
const unsigned int grid_size=16;					////assuming particles are initialized on a background grid
const unsigned int particle_n=pow(grid_size,3);	////assuming each grid cell has one particle at the beginning

__host__ void Test_N_Body_Simulation()
{
	////initialize position, velocity, acceleration, and mass
	
	double* pos_x=new double[particle_n];
	double* pos_y=new double[particle_n];
	double* pos_z=new double[particle_n];
	////initialize particle positions as the cell centers on a background grid
	double dx=1.0/(double)grid_size;
	for(unsigned int k=0;k<grid_size;k++){
		for(unsigned int j=0;j<grid_size;j++){
			for(unsigned int i=0;i<grid_size;i++){
				unsigned int index=k*grid_size*grid_size+j*grid_size+i;
				pos_x[index]=dx*(double)i;
				pos_y[index]=dx*(double)j;
				pos_z[index]=dx*(double)k;
			}
		}
	}

	double* vel_x=new double[particle_n];
	memset(vel_x,0x00,particle_n*sizeof(double));
	double* vel_y=new double[particle_n];
	memset(vel_y,0x00,particle_n*sizeof(double));
	double* vel_z=new double[particle_n];
	memset(vel_z,0x00,particle_n*sizeof(double));

	double* acl_x=new double[particle_n];
	memset(acl_x,0x00,particle_n*sizeof(double));
	double* acl_y=new double[particle_n];
	memset(acl_y,0x00,particle_n*sizeof(double));
	double* acl_z=new double[particle_n];
	memset(acl_z,0x00,particle_n*sizeof(double));

	double* mass=new double[particle_n];
	for(int i=0;i<particle_n;i++){
		mass[i]=100.0;
	}


	//////////////////////////////////////////////////////////////////////////
	////Default implementation: n-body simulation on CPU
	////Comment the CPU implementation out when you test large-scale examples
	auto cpu_start=chrono::system_clock::now();
	cout<<"Total number of particles: "<<particle_n<<endl;
	cout<<"Tracking the motion of particle "<<particle_n/2<<endl;
	// for(int i=0;i<time_step_num;i++){
	// 	N_Body_Simulation_CPU_Poorman(pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acl_x,acl_y,acl_z,mass,particle_n,dt,epsilon_squared);
	// 	cout<<"pos on timestep "<<i<<": "<<pos_x[particle_n/2]<<", "<<pos_y[particle_n/2]<<", "<<pos_z[particle_n/2]<<endl;
	// }
	auto cpu_end=chrono::system_clock::now();
	chrono::duration<double> cpu_time=cpu_end-cpu_start;
	cout<<"CPU runtime: "<<cpu_time.count()*1000.<<" ms."<<endl;

	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	////Your implementation: n-body simulator on GPU


	//////////////////////////////////////////////////////////////////////////
	////TODO 2: Your GPU functions are called here
	////Requirement: You need to copy data from the CPU arrays, conduct computations on the GPU, and copy the values back from GPU to CPU
	////The final positions should be stored in the same place as the CPU n-body function, i.e., pos_x, pos_y, pos_z
	////The correctness of your simulation will be evaluated by comparing the results (positions) with the results calculated by the default CPU implementations

	//////////////////////////////////////////////////////////////////////////

	for(unsigned int k=0;k<grid_size;k++){
		for(unsigned int j=0;j<grid_size;j++){
			for(unsigned int i=0;i<grid_size;i++){
				unsigned int index=k*grid_size*grid_size+j*grid_size+i;
				pos_x[index]=dx*(double)i;
				pos_y[index]=dx*(double)j;
				pos_z[index]=dx*(double)k;
			}
		}
	}

	
	memset(vel_x,0x00,particle_n*sizeof(double));
	
	memset(vel_y,0x00,particle_n*sizeof(double));
	
	memset(vel_z,0x00,particle_n*sizeof(double));

	
	memset(acl_x,0x00,particle_n*sizeof(double));
	
	memset(acl_y,0x00,particle_n*sizeof(double));
	
	memset(acl_z,0x00,particle_n*sizeof(double));

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	double *pos_x_dev;
	double *pos_y_dev;
	double *pos_z_dev;
	
	double *vel_x_dev;
	double *vel_y_dev;
	double *vel_z_dev;
	
	double *acl_x_dev;
	double *acl_y_dev;
	double *acl_z_dev;

	double *pos_x_dev_2;
	double *pos_y_dev_2;
	double *pos_z_dev_2;

	double *vel_x_dev_2;
	double *vel_y_dev_2;
	double *vel_z_dev_2;
	
	double *acl_x_dev_2;
	double *acl_y_dev_2;
	double *acl_z_dev_2;

	double *m_dev;
	double *dt_dev;
	double *epsilon_squared_dev;
	int *n_dev;

	cudaMalloc((void**)&pos_x_dev, particle_n * sizeof(double));
	cudaMalloc((void**)&pos_y_dev, particle_n * sizeof(double));
	cudaMalloc((void**)&pos_z_dev, particle_n * sizeof(double));

	cudaMalloc((void**)&vel_x_dev, particle_n * sizeof(double));
	cudaMalloc((void**)&vel_y_dev, particle_n * sizeof(double));
	cudaMalloc((void**)&vel_z_dev, particle_n * sizeof(double));

	cudaMalloc((void**)&acl_x_dev, particle_n * sizeof(double));
	cudaMalloc((void**)&acl_y_dev, particle_n * sizeof(double));
	cudaMalloc((void**)&acl_z_dev, particle_n * sizeof(double));

	cudaMalloc((void**)&pos_x_dev_2, particle_n * sizeof(double));
	cudaMalloc((void**)&pos_y_dev_2, particle_n * sizeof(double));
	cudaMalloc((void**)&pos_z_dev_2, particle_n * sizeof(double));

	cudaMalloc((void**)&vel_x_dev_2, particle_n * sizeof(double));
	cudaMalloc((void**)&vel_y_dev_2, particle_n * sizeof(double));
	cudaMalloc((void**)&vel_z_dev_2, particle_n * sizeof(double));

	cudaMalloc((void**)&acl_x_dev_2, particle_n * sizeof(double));
	cudaMalloc((void**)&acl_y_dev_2, particle_n * sizeof(double));
	cudaMalloc((void**)&acl_z_dev_2, particle_n * sizeof(double));

	cudaMalloc((void**)&m_dev, particle_n * sizeof(double));
	cudaMalloc((void**)&dt_dev, sizeof(double));
	cudaMalloc((void**)&epsilon_squared_dev, sizeof(double));
	cudaMalloc((void**)&n_dev, sizeof(int)); 


	cudaMemcpy(pos_x_dev, pos_x, particle_n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(pos_y_dev, pos_y, particle_n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(pos_z_dev, pos_z, particle_n * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(vel_x_dev, vel_x, particle_n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(vel_y_dev, vel_y, particle_n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(vel_z_dev, vel_z, particle_n * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(acl_x_dev, acl_x, particle_n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(acl_y_dev, acl_y, particle_n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(acl_z_dev, acl_z, particle_n * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemset(pos_x_dev_2, 0, particle_n * sizeof(double));
	cudaMemset(pos_y_dev_2, 0, particle_n * sizeof(double));
	cudaMemset(pos_z_dev_2, 0, particle_n * sizeof(double));

	cudaMemset(vel_x_dev_2, 0, particle_n * sizeof(double));
	cudaMemset(vel_y_dev_2, 0, particle_n * sizeof(double));
	cudaMemset(vel_z_dev_2, 0, particle_n * sizeof(double));

	cudaMemset(acl_x_dev_2, 0, particle_n * sizeof(double));
	cudaMemset(acl_y_dev_2, 0, particle_n * sizeof(double));
	cudaMemset(acl_z_dev_2, 0, particle_n * sizeof(double));

	cudaMemcpy(m_dev, mass, particle_n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dt_dev, &dt, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(epsilon_squared_dev,&epsilon_squared, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(n_dev, &particle_n, sizeof(int), cudaMemcpyHostToDevice);




	cout<<"Total number of particles: "<<particle_n<<endl;
	cout<<"Tracking the motion of particle "<<particle_n/2<<endl;
	for (int i = 0;i < time_step_num; i++)
	{
		if(i % 2 == 0)
		{
			N_BODY_GPU<<<dim3(particle_n/BDIMX), BDIMX>>>(
				pos_x_dev, pos_y_dev, pos_z_dev,
				vel_x_dev, vel_y_dev, vel_z_dev,
				acl_x_dev, acl_y_dev, acl_z_dev,
				pos_x_dev_2, pos_y_dev_2, pos_z_dev_2,
				vel_x_dev_2, vel_y_dev_2, vel_z_dev_2,
				acl_x_dev_2, acl_y_dev_2, acl_z_dev_2,
				m_dev,
				n_dev,
				dt_dev,
				epsilon_squared_dev);
			
	
			cudaMemcpy(pos_x, pos_x_dev_2, particle_n * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(pos_y, pos_y_dev_2, particle_n * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(pos_z, pos_z_dev_2, particle_n * sizeof(double), cudaMemcpyDeviceToHost);
	
			cout<<"pos on timestep "<<i<<": "<<pos_x[particle_n/2]<<", "<<pos_y[particle_n/2]<<", "<<pos_z[particle_n/2]<<endl;

		}
		else
		{
			N_BODY_GPU<<<dim3(particle_n/BDIMX), BDIMX>>>(
				pos_x_dev_2, pos_y_dev_2, pos_z_dev_2,
				vel_x_dev_2, vel_y_dev_2, vel_z_dev_2,
				acl_x_dev_2, acl_y_dev_2, acl_z_dev_2,

				pos_x_dev, pos_y_dev, pos_z_dev,
				vel_x_dev, vel_y_dev, vel_z_dev,
				acl_x_dev, acl_y_dev, acl_z_dev,

				m_dev,
				n_dev,
				dt_dev,
				epsilon_squared_dev);
			
	
			cudaMemcpy(pos_x, pos_x_dev, particle_n * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(pos_y, pos_y_dev, particle_n * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(pos_z, pos_z_dev, particle_n * sizeof(double), cudaMemcpyDeviceToHost);
	
			cout<<"pos on timestep "<<i<<": "<<pos_x[particle_n/2]<<", "<<pos_y[particle_n/2]<<", "<<pos_z[particle_n/2]<<endl;

		}
		// pos on timestep 0: 0.655429, 0.655429, 0.473758
		// pos on timestep 1: -0.407697, -0.407697, -0.144373
		// pos on timestep 2: -1.30937, -1.30937, -0.876057
		// pos on timestep 3: -2.19476, -2.19476, -1.62789
		// pos on timestep 4: -3.07345, -3.07345, -2.38426
		// pos on timestep 5: -3.94812, -3.94812, -3.14275
		// pos on timestep 6: -4.82021, -4.82021, -3.90248
		// pos on timestep 7: -5.69054, -5.69054, -4.663
		// pos on timestep 8: -6.55961, -6.55961, -5.42408
		// pos on timestep 9: -7.42775, -7.42775, -6.18555


	}


	cudaFree(pos_x_dev);
	cudaFree(pos_y_dev);
	cudaFree(pos_z_dev);

	cudaFree(vel_x_dev);
	cudaFree(vel_y_dev);
	cudaFree(vel_z_dev);
	
	cudaFree(acl_x_dev);
	cudaFree(acl_y_dev);
	cudaFree(acl_z_dev);

	cudaFree(pos_x_dev_2);
	cudaFree(pos_y_dev_2);
	cudaFree(pos_z_dev_2);

	cudaFree(vel_x_dev_2);
	cudaFree(vel_y_dev_2);
	cudaFree(vel_z_dev_2);
	
	cudaFree(acl_x_dev_2);
	cudaFree(acl_y_dev_2);
	cudaFree(acl_z_dev_2);

	cudaFree(m_dev);
	cudaFree(dt_dev);
	cudaFree(epsilon_squared_dev);
	cudaFree(n_dev);
	
	

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	//////////////////////////////////////////////////////////////////////////
	out<<"R0: "<<pos_x[particle_n/2]<<" " <<pos_y[particle_n/2]<<" " <<pos_z[particle_n/2]<<endl;
	out<<"T1: "<<gpu_time<<endl;

}

int main()
{
	if(name::team=="Team_X"){
		printf("\nPlease specify your team name and team member names in name::team and name::author to start.\n");
		return 0;
	}

	std::string file_name=name::team+"_competition_2_nbody.dat";
	out.open(file_name.c_str());
	
	if(out.fail()){
		printf("\ncannot open file %s to record results\n",file_name.c_str());
		return 0;
	}

	Test_N_Body_Simulation();

	return 0;
}
