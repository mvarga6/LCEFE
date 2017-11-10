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
		cout << endl << "\t" << item->AsString();
		break;
		 
	case LogEntryPriority::DEBUG:
		cout << endl << "[ DEBUG ] " << item->AsString();
		break;
		
	case LogEntryPriority::INFO:
		cout << endl << "[ INFO ] " << item->AsString();
		break;
		
	case LogEntryPriority::WARNING:
		cout << endl << "[ ** WARNING ** ] " << item->AsString();
		break;
		
	case LogEntryPriority::CRITICAL:
		cout << endl << endl << "[ *** CRITICAL *** ] " << item->AsString() << endl;
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
		cout << endl <<  "\t" << message;
		break;
		 
	case LogEntryPriority::DEBUG:
		cout << endl << "[ DEBUG ] " << message;
		break;
		
	case LogEntryPriority::INFO:
		cout << endl << "[ INFO ] " << message;
		break;
		
	case LogEntryPriority::WARNING:
		cout << endl << "[ ** WARNING ** ] " << message;
		break;
		
	case LogEntryPriority::CRITICAL:
		cout << endl << endl << "[ *** CRITICAL *** ] " << message << endl;
		break;
	}
}

void ConsoleLogger::StaticMsg(const string& message)
{
	cout << "\r" << "[ INFO ] " << message << "\t";
}


void ConsoleLogger::Error(const string& message)
{
	cout << endl << "[ ERROR ] " << message;
}
