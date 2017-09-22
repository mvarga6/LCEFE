ext:
	cd extlib/jsmn && $(MAKE) && $(MAKE) test
	cd extlib/gmsh_io && $(MAKE) all

all: ext main.cu clparse.cpp genrand.cpp parameters_reader.cpp parameters_writer.cpp
	./build
	
clean:
	cd extlib/jsmn && $(MAKE) clean
	cd extlib/gmsh_io && $(MAKE) clean
	cd builds && rm *
	
#clean vtk: 
#	rm VTKOUT/*
