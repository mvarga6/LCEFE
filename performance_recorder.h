#ifndef __PERFORMANCE_RECORDER_H__
#define __PERFORMANCE_RECORDER_H__

#include <map>
#include <string>

using namespace std;

struct PerformanceMetric
{
	cudaEvent_t start;
	cudaEvent_t stop;
	float elapsed;
	float elapsed_total;
	float elapsed_average;
	float elapsed_min;
	float elapsed_max;
	size_t nevents;
	bool recording;
	
	PerformanceMetric();
	
	void Start();
	void Mark(bool restart);
	void Stop();
	
	string AsString();
};

class PerformanceRecorder
{
	map<string, PerformanceMetric> metrics;

public:
	PerformanceMetric* Create(string key);
	PerformanceMetric* Start(string key);
	PerformanceMetric* Mark(string key);
	PerformanceMetric* Stop(string key);
	void Log(string key);
};

#endif
