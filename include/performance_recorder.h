#ifndef __PERFORMANCE_RECORDER_H__
#define __PERFORMANCE_RECORDER_H__

#include <map>
#include <string>
#include "logger.h"
#include "defines.h"

using namespace std;

struct PerformanceMetric
{
	cudaEvent_t start;
	cudaEvent_t stop;
	real elapsed;
	real elapsed_total;
	real elapsed_average;
	real elapsed_min;
	real elapsed_max;
	size_t nevents;
	bool recording;
	string name;
	
	PerformanceMetric();
	PerformanceMetric(string name);
	PerformanceMetric(const PerformanceMetric& copy);
	
	PerformanceMetric& operator=(const PerformanceMetric& rhs);
	
	void Start();
	void Mark(bool restart);
	void Stop();
	
	string AsString();
	void WriteToLogger(Logger *log);
};

class PerformanceRecorder
{
	map<string, PerformanceMetric> metrics;
	Logger * log;

public:
	PerformanceRecorder(Logger *logger);

	PerformanceMetric* Create(string key);
	PerformanceMetric* Start(string key);
	PerformanceMetric* Mark(string key);
	PerformanceMetric* Stop(string key);
	void Log(string key);
};

#endif
