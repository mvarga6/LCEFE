#include "../include/output_writer.h"
#include <fstream>

using namespace std;

VtkWriter::VtkWriter(string outputBaseName)
{
	this->outputBase = outputBaseName;
	this->points_bound = false;
	this->cells_bound = false;
	this->cells_data_bound = false;
}


void VtkWriter::BindPoints(real *points, int nPoints, DataFormat format, int dim)
{
	this->_points = points;
	this->npoints = nPoints;
	this->points_format = format;
	this->points_dim = dim;
	this->points_bound = true;
}


void VtkWriter::BindCells(int *cells, int nCells, DataFormat format, CellType type)
{
	this->_cells = cells;
	this->ncells = nCells;
	this->cells_format = format;
	this->cells_type = type;
	this->cells_bound = true;
}


void VtkWriter::BindCellData(real *cellsData, int nCells, DataFormat format, int dim, string cellDataName)
{
	this->_cell_data = cellsData;
	this->ncellsdata = nCells;
	this->cells_data_format = format;
	this->cells_data_dim = dim;
	this->cells_data_name = cellDataName;
	this->cells_data_bound = true;
}


bool VtkWriter::Write()
{
	return false;
}


bool VtkWriter::Write(int fileNumber)
{
	// open output fileNumber
	string fileName = this->outputBase + "_" + to_string(fileNumber) + ".vtk";
	ofstream out(fileName, ios::out);

	if(!out.is_open())
	{
		return false;
	}

	// write the sections of the output file	
	WriteHeader(out, "Tetrahedral Mesh Visualization");
	
	if (this->points_bound)
	{
		WritePoints(out);
	}
	
	if (this->cells_bound) 
	{
		WriteCells(out);
	}
	
	if (this->cells_data_bound)
	{
		WriteCellData(out, this->cells_data_name);
	}
	
	out.close();
	return true;
}


void VtkWriter::WriteHeader(ofstream& out, string title)
{
	// print header
	out << "# vtk DataFile Version 3.1" << endl;
	out << title << endl;
	out << "ASCII" << endl;
	out << "DATASET UNSTRUCTURED_GRID" << endl;
	out << endl;
}


void VtkWriter::WritePoints(ofstream& out)
{
	// print title of section
	out << "POINTS " << this->npoints << " real" << endl;
	
	// print based on format
	switch(this->points_format)
	{	
	case DataFormat::LinearizedByDimension:
	
		// print each point
		for(int i = 0; i < this->npoints; i++)
		{
			// print value for each dimension of that point
			for (int d = 0; d < this->points_dim; d++)
			{
				out << this->_points[i + d*this->npoints] << " ";  
			}
			out << endl;
		}
		out << endl;
		
		break;
	
	case DataFormat::LinearizedByItem:
	
		// print each point
		for(int i = 0; i < this->npoints; i++)
		{
			// print value for each dimension of that point
			for (int d = 0; d < this->points_dim; d++)
			{
				out << this->_points[this->points_dim * i + d] << " ";  
			}
			out << endl;
		}
		out << endl;
	
		break;
	
	default:
		break;
	}
}


void VtkWriter::WriteCells(ofstream& out)
{
	// print section title (NAME num_of_cells num_of_ints)
	out << "CELLS " << this->ncells << " " 
		<< this->ncells * ((int)this->cells_type + 1) << endl;

	// print based on format
	switch(this->cells_format)
	{	
	case DataFormat::LinearizedByDimension:
	
		// print each point
		for(int i = 0; i < this->ncells; i++)
		{
			// print cells type (# of nodes per tet)
			out << (int)cells_type << " ";
		
			// print value for each dimension of that point
			for (int d = 0; d < (int)this->cells_type; d++)
			{
				out << this->_cells[i + d*this->ncells] << " ";  
			}
			out << endl;
		}
		out << endl;
		
		break;
	
	case DataFormat::LinearizedByItem:
	
		// print each point
		for(int i = 0; i < this->ncells; i++)
		{
			// print cells type (# of nodes per tet)
			out << (int)cells_type << " ";
		
			// print value for each dimension of that point
			for (int d = 0; d < (int)this->cells_type; d++)
			{
				out << this->_cells[(int)this->cells_type * i + d] << " ";  
			}
			out << endl;
		}
		out << endl;
		
		break;
	
	default:
		break;
	}
	
	// cell type section header
	out << "CELL_TYPES " << this->ncells << endl;
	
	// print VTK cell types (10 for tetrahedra)
	for (int i = 0; i < this->ncells; i ++)
	{
		out << 10 << endl;
	}
	out << endl;
}


void VtkWriter::WriteCellData(ofstream& out, string title)
{
	// print section title
	out << "CELL_DATA " << this->ncellsdata << endl;
	out << "SCALARS " << title << " real " << this->cells_data_dim << endl;
	out << "LOOKUP_TABLE default" << endl;
	
	
	// print based on format
	switch(this->cells_data_format)
	{	
	case DataFormat::LinearizedByDimension:
	
		// print each point
		for(int i = 0; i < this->ncellsdata; i++)
		{
			// print value for each dimension of that point
			for (int d = 0; d < this->cells_data_dim; d++)
			{
				out << this->_cell_data[i + d*this->ncellsdata] << " ";  
			}
			out << endl;
		}
		out << endl;
		
		break;
	
	case DataFormat::LinearizedByItem:
	
		// print each point
		for(int i = 0; i < this->ncellsdata; i++)
		{
			// print value for each dimension of that point
			for (int d = 0; d < this->cells_data_dim; d++)
			{
				out << this->_cell_data[this->cells_data_dim * i + d] << " ";  
			}
			out << endl;
		}
		out << endl;
		
		break;
	
	default:
		break;
	}
}
















