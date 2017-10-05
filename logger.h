#ifndef __LOGGER_H__
#define __LOGGER_H__

#include <string>

using namespace std;

enum class LoggerType : int
{
	NONE = 0,
	CONSOLE = 1,
	FILE = 2
};

inline LoggerType ConvertToLoggerType(int t)
{
	switch(t)
	{
	case 0: return LoggerType::NONE;
	case 1: return LoggerType::CONSOLE;
	case 2: return LoggerType::FILE;
	default: return LoggerType::CONSOLE;
	}
}

/*
  The priority of a log item
*/
enum class LogEntryPriority : int
{
	VERBOSE = 4,
	DEBUG = 3,
	INFO = 2,
	WARNING = 1,
	CRITICAL = 0
};

/*
  Helper to safely convert from int to LogEntryPriority
*/
inline LogEntryPriority ConvertToLogEntryPriority(int p)
{
	switch(p)
	{
	case 0: return LogEntryPriority::CRITICAL;
	case 1: return LogEntryPriority::WARNING;
	case 2: return LogEntryPriority::INFO;
	case 3: return LogEntryPriority::DEBUG;
	case 4: return LogEntryPriority::VERBOSE;
	default: return LogEntryPriority::INFO;
	}
}

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
	ErrorLog(string, const exception&, LogEntryPriority);
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
