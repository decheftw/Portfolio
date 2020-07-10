//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League code test
//// TEAM NAME: dns
//// Team Members: Gao Chen, Nicolas Flores, Shikhar Sinha
//////////////////////////////////////////////////////////////////////////
#ifndef __Round_0_h__
#define __Round_0_h__

#include <chrono>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
using namespace std;

//////////////////////////////////////////////////////////////////////////
////TODO 0: Please replace the following strings with your team name and author names
////Note: Please do not use space in the string, use "_" instead
//////////////////////////////////////////////////////////////////////////

namespace name
{
std::string team = "dns";
std::string author_1 = "Gao_Chen";
std::string author_2 = "Nicolas_Flores";
std::string author_3 = "Shikhar_Sinha"; ////optional
};										// namespace name

//////////////////////////////////////////////////////////////////////////
////TODO 1: Please replace the following code to calculate the sum of an integer array
////with your own implementation to get better performance
//////////////////////////////////////////////////////////////////////////

int Int_Vector_Sum(const int *array, const int size)
{
	int s1 = 0;
	int s2 = 0;
	int s3 = 0;
	int s4 = 0;

	for (int i = 0; i < size; i += 4)
	{
		s1 += array[i];
		s2 += array[i + 1];
		s3 += array[i + 2];
		s4 += array[i + 3];
	}
	return s1 + s2 + s3 + s4;
}

//////////////////////////////////////////////////////////////////////////
////TODO 2: Please replace the following code to calculate the sum of a double array
////with your own implementation to get better performance
//////////////////////////////////////////////////////////////////////////

double Double_Vector_Sum(const double *array, const int size)
{
	double s1 = 0;
	double s2 = 0;
	double s3 = 0;
	double s4 = 0;
	for (int i = 0; i < size; i += 4)
	{
		s1 += array[i];
		s2 += array[i + 1];
		s3 += array[i + 2];
		s4 += array[i + 3];
	}
	return s1 + s2 + s3 + s4;
}

#endif
