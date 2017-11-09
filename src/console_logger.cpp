#include "logger.h"
#include <sstream>
#include <iostream>

using namespace std;

/*
  ErrorLog
*/
ErrorLog::ErrorLog(string message, LogEntryPriority priority = LogEntryPriority::INFO)
{
	this->priority = priority;
	this->msg = message;
}

ErrorLog::ErrorLog(string message, exception& e, LogEntryPriority priority = LogEntryPriority::INFO)
	: ErrorLog(message, priority)
{
	this->e = e;
}

string ErrorLog::AsString()
{
	stringstream ss;
	ss << "ErrorMsg: " << msg << " Exception: " << e.what();
	return ss.str();	
}


Logger * Logger::Default = new ConsoleLogger(LogEntryPriority::VERBOSE);


/*
  ConsoleLogger
*/
ConsoleLogger::ConsoleLogger(LogEntryPriority priority)
{
	this->min_priority = priority;
}


void ConsoleLogger::Log(LogEntry *item)
{
	if (item->priority < min_priority)
	{
		return;
	}

	switch(item->priority)
	{
	case LogEntryPriority::VERBOSE:
		cout << "\t" << item->AsString() << endl;
		break;
		 
	case LogEntryPriority::DEBUG:
		cout << "[ DEBUG ] " << item->AsString() << endl;
		break;
		
	case LogEntryPriority::INFO:
		cout << "[ INFO ] " << item->AsString() << endl;
		break;
		
	case LogEntryPriority::WARNING:
		cout << "[ WARNING ] " << item->AsString() << endl;
		break;
		
	case LogEntryPriority::CRITICAL:
		cout << endl << "[ CRITICAL ] " << item->AsString() << endl << endl;
		break;
	}
}

void ConsoleLogger::Msg(const string& message, LogEntryPriority priority)
{
	if (priority < min_priority)
	{
		return;
	}

	switch(priority)
	{
	case LogEntryPriority::VERBOSE:
		cout << "\t" << message << endl;
		break;
		 
	case LogEntryPriority::DEBUG:
		cout << "[ DEBUG ] " << message << endl;
		break;
		
	case LogEntryPriority::INFO:
		cout << "[ INFO ] " << message << endl;
		break;
		
	case LogEntryPriority::WARNING:
		cout << "[** WARNING **] " << message << endl;
		break;
		
	case LogEntryPriority::CRITICAL:
		cout << endl << "[*** CRITICAL ***] " << message << endl << endl;
		break;
	}
}

void ConsoleLogger::StaticMsg(const string& message)
{
	cout << "\r" << "[ INFO ] " << message << "\t";
}

