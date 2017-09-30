#include "performance_recorder.h"
#include "errorhandle.h"
#include <sstream>

using namespace std;

PerformanceMetric::PerformanceMetric()
{
	elapsed = elapsed_average = elapsed_total = 0.0f;
	elapsed_max = 0;
	elapsed_min = 100000000;
	nevents = 0;
	recording = false;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
}


void PerformanceMetric::Start()
{
	HANDLE_ERROR(cudaEventRecord(start));
	recording = true;
}


void PerformanceMetric::Mark(bool restart = true)
{
	// start if we haven't yet
	if (!recording)
	{
		return;
	}
	
	// Update the metric
	nevents++;

	float _elapsed;
	HANDLE_ERROR(cudaEventRecord(stop));
	HANDLE_ERROR(cudaEventSynchronize(stop));		
	HANDLE_ERROR(cudaEventElapsedTime(&_elapsed, 
		start, stop));
			
	elapsed = _elapsed;
	elapsed_total += elapsed;
	elapsed_average = elapsed_total / (float)nevents;
		
	if (elapsed < elapsed_min)
	{
		elapsed_min = elapsed;
	}
		
	if (elapsed > elapsed_max)
	{
		elapsed_max = elapsed;
	}
		
	// reset the start events
	if (restart)
	{
		HANDLE_ERROR(cudaEventRecord(start));
	}
}


void PerformanceMetric::Stop()
{
	if (recording)
	{
		Mark(false);
		recording = false;
	}
}

string PerformanceMetric::AsString()
{
	stringstream ss;
	ss << " - n-events:\t" << nevents << endl;
	ss << " - last:\t" << elapsed << " ms"<< endl;
	ss << " - ave:\t\t" << elapsed_average << " ms" << endl;
	ss << " - total:\t" << elapsed_total / 1000.0f << " sec"<< endl;
	ss << " - min:\t\t" << elapsed_min << " ms" << endl;
	ss << " - max:\t\t" << elapsed_max << " ms" << endl;
	return ss.str();
}

PerformanceMetric* PerformanceRecorder::Create(string key)
{
	// already exists
	auto it = metrics.find(key); 
	if ( it == metrics.end() )
	{
		// add a new performance metric
		PerformanceMetric metric;
		metrics[key] = metric;
	}
	return &metrics[key];
}


PerformanceMetric* PerformanceRecorder::Start(string key)
{
	auto it = metrics.find(key);
	if (it != metrics.end())
	{
		 it->second.Start();
		 return &metrics[key]; 
	}
	return NULL;
}


PerformanceMetric* PerformanceRecorder::Mark(string key)
{
	auto it = metrics.find(key);
	if (it != metrics.end())
	{
		it->second.Mark();
		return &metrics[key];
	}
	return NULL;
}

PerformanceMetric* PerformanceRecorder::Stop(string key)
{
	auto it = metrics.find(key);
	if (it != metrics.end())
	{
		it->second.Stop();
		return &metrics[key];
	}
	return NULL;
}


void PerformanceRecorder::Log(string key)
{
	auto it = metrics.find(key);
	if (it != metrics.end())
	{
		printf("\nStats for \"%s\" metric\n%s", key.c_str() ,it->second.AsString().c_str());
	}
}

