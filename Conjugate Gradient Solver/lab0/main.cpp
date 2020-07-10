//////////////////////////////////////////////////////////////////////////
////GPU Computing Premier League
////Round 0: C++ code test
//////////////////////////////////////////////////////////////////////////

////Note: You do not need to modify anything in the main file

#include <chrono>
#include <fstream>
#include "gpu_league_round_0.h"

int main()
{
	std::string file_name=name::team+"_round_0.dat";
	ofstream out(file_name);
	if(!out){
		cout<<"cannot open file "<<file_name<<endl; 
		return 0;
	}
	
	double avg_time_1=(double)0;
	int result_1=0;
	for(int i=0;i<100;i++){
		int size=10000;
		std::vector<int> array(10000,1);
		auto start=chrono::system_clock::now();

		result_1=Int_Vector_Sum(&array[0],size);
	
		auto end=chrono::system_clock::now();
		chrono::duration<double> t=end-start;
		double time=t.count()*1000.;	
		cout<<"result for int sum: "<<result_1<<endl;
		cout<<"run time for int sum: "<<time<<" ms."<<endl;
		avg_time_1+=time;
	}
	avg_time_1/=(double)100;
	out<<"R1: "<<result_1<<endl;
	out<<"T1: "<<avg_time_1<<endl;

	double avg_time_2=(double)0;
	double result_2=(double)0;
	for(int i=0;i<100;i++){
		int size=10000;
		std::vector<double> array(10000,1);
		auto start=chrono::system_clock::now();

		result_2=Double_Vector_Sum(&array[0],size);
	
		auto end=chrono::system_clock::now();
		chrono::duration<double> t=end-start;
		double time=t.count()*1000.;	
		cout<<"result for double sum: "<<result_2<<endl;
		cout<<"run time for double sum: "<<time<<" ms."<<endl;
		avg_time_2+=time;
	}
	avg_time_2/=(double)100;
	out<<"R2: "<<result_2<<endl;
	out<<"T2: "<<avg_time_2<<endl;

	cout<<"\n\n--------------------------------"<<endl;
	cout<<"R1: "<<result_1<<endl;
	cout<<"T1: "<<avg_time_1<<endl;
	cout<<"R2: "<<result_2<<endl;
	cout<<"T2: "<<avg_time_2<<endl;
	cout<<"--------------------------------"<<endl;
	out.close();
	return 0;
}
