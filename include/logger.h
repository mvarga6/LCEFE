#ifndef __LOGGER_H__
#define __LOGGER_H__

#include <string>
#include <sstream>
#include <cstdarg>
#include <memory>

using namespace std;

/**
 * The priority of a log item
 */
enum class LogEntryPriority : int
{
	VERBOSE = 0,
	DEBUG = 1,
	INFO = 2,
	WARNING = 3,
	CRITICAL = 4
};


/**
 * Abstact parent for anything that can be logged
 */
class LogEntry
{
public:

	///
	/// The priority for this item
	LogEntryPriority priority;

	///
	/// Get the item as a string
	virtual string AsString() = 0;
};


/**
 * A log entry type for logging errors
 */
class ErrorLog : public LogEntry
{
	///
	/// The error message
	string msg;

	///
	/// an exception to go along with the error
	exception e;
	
public:
	ErrorLog(string, LogEntryPriority);
	ErrorLog(string, exception&, LogEntryPriority);
	string AsString();	
};

/**
 * 	A helper function to make formated std::strings easily
 */
 string formatted(const char *const format, ...)
   __attribute__ ((formatted (printf, 1, 2)));
//string formatted(const char* format, ...);
//string formatted(const char* fomrat, va_list args);

/**
 * Abstract parent of anything that is a logger
 */
class Logger
{
public:
	virtual void Log(LogEntry*) = 0;
	virtual void Msg(const string& message, LogEntryPriority priority = LogEntryPriority::INFO) = 0;
	virtual void StaticMsg(const string& message) = 0;
	//virtual void Error(const string& message) = 0;

	///
	/// Formatted, explicit priority functions
	///
	virtual void Detail(const string&) = 0;
	virtual void Debug(const string&) = 0;
	virtual void Info(const string&) = 0;
	virtual void Warning(const string&) = 0;
	virtual void Error(const string&) = 0;

	static Logger * Default;

protected:
	static constexpr const char* err_label = "[ *** ERROR *** ]";
	static constexpr const char* warn_label = "[ ** WARNING ** ]";
	static constexpr const char* info_label = "[ INFO ]";
	static constexpr const char* debug_label = "[ DEBUG ]";
	static constexpr const char* verb_label = "";
};

/**
 * Log things to the console
 */
class ConsoleLogger : public Logger
{
	LogEntryPriority min_priority;
public:
	ConsoleLogger(LogEntryPriority priority = LogEntryPriority::INFO);
	void Log(LogEntry*);
	void Msg(const string& message, LogEntryPriority priority = LogEntryPriority::INFO);
	void StaticMsg(const string& message);
	//void Error(const string& message);

	void Detail(const string&);
	void Debug(const string&);
	void Info(const string&);
	void Warning(const string&);
	void Error(const string&);

private:
	void Print(const string& label, const string& message);
};

#endif
