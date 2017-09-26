ext:
	cd extlib/jsmn && $(MAKE) && $(MAKE) test
	cd extlib/gmsh_io && $(MAKE) all

all: ext main.cu clparse.cpp genrand.cpp parameters_reader.cpp parameters_writer.cpp
	./build
	
clean:
	cd VTKOUT && rm *.vtk || true && rm *.xyzv || true && rm *.dat || true
	cd builds && rm * || true
	cd extlib/jsmn && $(MAKE) clean
	cd extlib/gmsh_io && $(MAKE) clean
	
#clean vtk: 
#	rm VTKOUT/*
