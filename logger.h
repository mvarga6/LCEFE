#ifndef __LOGGER_H__
#define __LOGGER_H__

#include <string>

using namespace std;

/*
  The priority of a log item
*/
enum class LogEntryPriority : int
{
	VERBOSE = 0,
	DEBUG = 1,
	INFO = 2,
	WARNING = 3,
	CRITICAL = 4
};


/*
  Abstact parent for anything that can be logged
*/
class LogEntry
{
public:
	LogEntryPriority priority;
	virtual string AsString() = 0;
};


/*
  A log entry type for logging errors
*/
class ErrorLog : public LogEntry
{
	string msg;
	exception e;
	
public:
	ErrorLog(string, LogEntryPriority);
	ErrorLog(string, exception&, LogEntryPriority);
	string AsString();	
};


/*
  Abstract parent of anything that is a logger
*/
class Logger
{
public:
	virtual void Log(LogEntry*) = 0;
	virtual void Msg(const string& message, LogEntryPriority priority = LogEntryPriority::INFO) = 0;
	
	static Logger * Default;
};

/*
  Log things to the console
*/
class ConsoleLogger : public Logger
{
	LogEntryPriority min_priority;
public:
	ConsoleLogger(LogEntryPriority);
	void Log(LogEntry*);
	void Msg(const string& message, LogEntryPriority priority = LogEntryPriority::INFO);
};

#endif
