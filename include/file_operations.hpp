#ifndef __FILE_OPERATIONS_HPP__
#define __FILE_OPERATIONS_HPP__

#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;

///
/// Structure to store commmon information about a local file
struct FileInfo
{
	///
	/// Name of the file
	string FileName;

	///
	/// Name of the file excluding its extension
	string FileNameNoExt;

	///
	/// If existing, the path to the file excluding
	/// taken from the front of the full file name/path
	string Path;

	///
	/// Just the extension of the file
	string Extension;
	
	FileInfo();
};

///
/// Service class for operation on files
class FileOperations
{
public:
	static FileInfo GetFileInfo(const string&);
	static bool Exists(const string&);
};

#endif
