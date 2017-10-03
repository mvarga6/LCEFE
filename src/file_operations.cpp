#include "file_operations.hpp"

using namespace std;

FileInfo::FileInfo()
{
	FileName = "";
	FileNameNoExt = "";
	Path = "";
	Extension = "";
}

FileInfo FileOperations::GetFileInfo(const std::string& str)
{
	FileInfo info;

	// check if str is empty 
	if (str.empty())
	{
		return info;
	}

	// get to file name
	std::size_t dir_pos = str.find_last_of("/\\");
	std::size_t ext_pos = str.find_last_of(".");
	
	//
	// Choose the case
	//
	
	// get the extension if it exists
	if (ext_pos != string::npos && dir_pos != string::npos)
	{
		// last '.' to end
		info.Extension = str.substr(ext_pos + 1);
		
		// last '\\' or '/' to end of string (includes extension)
		info.FileName = str.substr(dir_pos + 1);
		
		// from beginning to last '\\' or '/'
		info.Path = str.substr(0, dir_pos);
		
		// remove extension from filename
		info.FileNameNoExt = str.substr(dir_pos, ext_pos + 1);
	}
	else if (ext_pos != string::npos && dir_pos == string::npos)
	{
		// last '.' to end
		info.Extension = str.substr(ext_pos + 1);
		
		// the string is just a file name w/ ext
		info.FileName = str;
		
		// get the base of filename
		info.FileNameNoExt = str.substr(0, ext_pos + 1);
	}
	else if (ext_pos == string::npos && dir_pos != string::npos)
	{
		// the string is just a file name w/ ext
		info.FileName = str.substr(dir_pos);
		
		// from begin to last '\\' or '/'
		info.Path = str.substr(0, dir_pos + 1);
		
		// we don't have an extension already
		info.FileNameNoExt = info.FileName;
	}
	else // no extension or directory
	{
		info.FileName = str;
		info.FileNameNoExt = str;
	}
	
	return info;
}


bool FileOperations::Exists(const string &filename)
{
	struct stat buffer;   
	return (stat (filename.c_str(), &buffer) == 0);
}




