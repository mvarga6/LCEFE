#include "../include/logger.h"
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
		cout << endl << verb_label << " " << item->AsString();
		break;
		 
	case LogEntryPriority::DEBUG:
		cout << endl << debug_label << " " << item->AsString();
		break;
		
	case LogEntryPriority::INFO:
		cout << endl << info_label << " " << item->AsString();
		break;
		
	case LogEntryPriority::WARNING:
		cout << endl << warn_label << " " << item->AsString();
		break;
		
	case LogEntryPriority::CRITICAL:
		cout << endl << endl << err_label << " " << item->AsString() << endl;
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
		cout << endl <<  verb_label << " " << message;
		break;
		 
	case LogEntryPriority::DEBUG:
		cout << endl << debug_label << " " << message;
		break;
		
	case LogEntryPriority::INFO:
		cout << endl << info_label << " " << message;
		break;
		
	case LogEntryPriority::WARNING:
		cout << endl << warn_label << " " << message;
		break;
		
	case LogEntryPriority::CRITICAL:
		cout << endl << endl << err_label << " " << message << endl;
		break;
	}
}

void ConsoleLogger::StaticMsg(const string& message)
{
	cout << "\r" << info_label << " " << message << "\t";
}


// void ConsoleLogger::Error(const string& message)
// {
// 	cout << endl << err_label << " " << message;
// }


void ConsoleLogger::Detail(const string& message)
{
	Print(verb_label, message);
}

void ConsoleLogger::Debug(const string& message)
{
	Print(debug_label, message);
}

void ConsoleLogger::Info(const string& message)
{
	Print(info_label, message);
}

void ConsoleLogger::Warning(const string& message)
{
	Print(warn_label, message);
}

void ConsoleLogger::Error(const string& message)
{
	Print(err_label, message);
}

void ConsoleLogger::Print(const string& label, const string& message)
{
	cout << endl << label << " " << message;
}