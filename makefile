
#OBJECTS = clparse.o genrand.o physics_model.o output_writer.o parameters_reader.o parameters_writer.o main.o 
EXTLIB = extlib/gmsh_io/libgmsh_io.a extlib/jsmn/libjsmn.a
FLAGS = -lcurand -std=c++11 -Wno-deprecated-gpu-targets

CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))



#all: ext main.cu clparse.cpp genrand.cpp physics_model.cpp parameters_reader.cpp parameters_writer.cpp
#	./build
	
#build: $(OBJECTS)
all: ext $(OBJ_FILES)
	nvcc $(FLAGS) $(OBJ_FILES) $(EXTLIB) -o builds/gafe6
	
obj/%.o: src/%.cpp
	nvcc -x cu $(FLAGS) -I. -dc $< -o $@
	
ext:
	cd extlib/jsmn && $(MAKE) && $(MAKE) test
	cd extlib/gmsh_io && $(MAKE) all
		
clean:
	rm -f *.o || true
	cd VTKOUT && rm *.vtk || true && rm *.xyzv || true && rm *.dat || true
	cd builds && rm * || true
	rm obj/*.o || true
	cd extlib/jsmn && $(MAKE) clean
	cd extlib/gmsh_io && $(MAKE) clean
	
#clean vtk: 
#	rm VTKOUT/*
