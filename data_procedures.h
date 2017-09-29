#ifndef __DATA_PROCEDURES_H__
#define __DATA_PROCEDURES_H__

#include <vector>
#include "data_operations.h"
using namespace std;
/*
  A storage container for list of DataOperations, logically 
  grouped and ordered into a Procedure.  
  
  For custom precudures, define a new DataProcedure and
  manually add DataOperations to the Operations list.
*/
class DataProcedure { public: vector<DataOperation*> Operations; };

/*
  Some out-of-the-box implementations of a DataProcedure 
  to do some things that are always required.
*/
class GetPrintData : public DataProcedure { public: GetPrintData(); };
class PushAllToGpu : public DataProcedure { public: PushAllToGpu(); };

#endif
