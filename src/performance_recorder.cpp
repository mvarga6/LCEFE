#include "performance_recorder.h"
#include "errorhandle.h"
#include <sstream>

using namespace std;

PerformanceMetric::PerformanceMetric()
{
}

PerformanceMetric::PerformanceMetric(string name)
{
	this->elapsed = elapsed_average = elapsed_total = 0.0f;
	this->elapsed_max = 0;
	this->elapsed_min = 100000000;
	this->nevents = 0;
	this->recording = false;
	this->name = name;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
}

PerformanceMetric::PerformanceMetric(const PerformanceMetric& copy)
{
	this->start = copy.start;
	this->stop = copy.stop;
	this->elapsed = copy.elapsed;
	this->elapsed_total = copy.elapsed_total;
	this->elapsed_average = copy.elapsed_average;
	this->elapsed_min = copy.elapsed_min;
	this->elapsed_max = copy.elapsed_max;
	this->nevents = copy.nevents;
	this->recording = copy.recording;
	this->name = copy.name;
}

PerformanceMetric& PerformanceMetric::operator=(const PerformanceMetric& rhs)
{
	this->start = rhs.start;
	this->stop = rhs.stop;
	this->elapsed = rhs.elapsed;
	this->elapsed_total = rhs.elapsed_total;
	this->elapsed_average = rhs.elapsed_average;
	this->elapsed_min = rhs.elapsed_min;
	this->elapsed_max = rhs.elapsed_max;
	this->nevents = rhs.nevents;
	this->recording = rhs.recording;
	this->name = rhs.name;
	return (*this);
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
	elapsed_average = elapsed_total / (real)nevents;

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

void PerformanceMetric::WriteToLogger(Logger *log)
{
	log->Info(formatted("Performance Metric %s", this->name.c_str()));
	log->Info(formatted("Event count:  %d", nevents));
	log->Info(formatted("Total:        %0.3fs", elapsed_total / (real)1000.0));
	log->Info(formatted("Min/Ave/Max:  %0.3fms / %0.3fms / %0.3fms", elapsed_min, elapsed_average, elapsed_max));
}

PerformanceRecorder::PerformanceRecorder(Logger * logger)
{
	this->log = logger;
}

PerformanceMetric* PerformanceRecorder::Create(string key)
{
	// already exists
	auto it = metrics.find(key);
	if ( it == metrics.end() )
	{
		// add a new performance metric
		PerformanceMetric metric(key);
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
		it->second.WriteToLogger(this->log);
	}
}

