#ifndef __OUTPUT_WRITER_H__
#define __OUTPUT_WRITER_H__


#include "datastruct.h"
#include <string>
#include <fstream>

using namespace std;

class OutputWriter
{
	public:
		virtual bool Write() = 0;
		virtual bool Write(int) = 0;
};

enum class DataFormat : int
{
	Linear = 0,
	LinearizedByDimension = 1,
	LinearizedByItem = 2,
};

enum class CellType : int
{
	Tetrahedral = 4
};

class VtkWriter : public OutputWriter
{
	// output base
	string outputBase;
	
	// points in outputfile
	float *_points;
	int npoints;
	DataFormat points_format;
	int points_dim;
	bool points_bound;
	
	// cells in output file
	int *_cells;
	int ncells;
	DataFormat cells_format;
	CellType cells_type;
	bool cells_bound;
	
	// cell data in output file
	float *_cell_data;
	int ncellsdata;
	DataFormat cells_data_format;
	int cells_data_dim;
	string cells_data_name;
	bool cells_data_bound;

	public:
	
		// Create writer with output base name
		VtkWriter(string outputBaseName);
		
		// bind data pointers
		void BindPoints(float *points, int nPoints, DataFormat format, int dim);
		void BindCells(int *cells, int nCells, DataFormat format, CellType type);
		void BindCellData(float *cellsData, int nCells, DataFormat format, int dim, string cellDataName);
		
		// write to output file
		bool Write();
		
		// write to output file x
		bool Write(int);
		
	private:
	
		void WriteHeader(ofstream& out, string title);
		void WritePoints(ofstream& out);
		void WriteCells(ofstream& out);
		void WriteCellData(ofstream& out, string title);
};


#endif
