#ifndef __FILE_OPERATIONS_HPP__
#define __FILE_OPERATIONS_HPP__

#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;

struct FileInfo
{
	string FileName;
	string FileNameNoExt;
	string Path;
	string Extension;
	
	FileInfo();
};

class FileOperations
{
public:
	static FileInfo GetFileInfo(const string&);
	static bool Exists(const string&);
};

#endif
