BUILDNAME = gafe6

# Path to external libraries
EXTLIB = extlib/gmsh_io/libgmsh_io.a extlib/jsmn/libjsmn.a

# Compiler Flags for Debug and Release
ifeq ($(DEBUG),true)
	FLAGS = -lcurand -ccbin=g++ -std=c++11 -G -g -lineinfo -Wno-deprecated-gpu-targets
	OBJDIR := obj/debug
	BLDDIR := builds/debug
else
	FLAGS = -lcurand -ccbin=g++ -std=c++11 -Xptxas -O3,-v -Wno-deprecated-gpu-targets
	OBJDIR := obj/release
	BLDDIR := builds/release
endif

# Directories in repo
SRCDIR := src
VTKDIR := VTKOUT
DOCDIR := doc
WORLD := $(OBJDIR) $(VTKDIR) $(BLDDIR)

# Source files
CPP_FILES := $(wildcard $(SRCDIR)/*.cpp)

# Generated files
OBJ_FILES := $(addprefix $(OBJDIR)/,$(notdir $(CPP_FILES:.cpp=.o)))
	
# Creates dirs, makes external deps, 
# compiles src files to obj then
# links into executable 
all: world ext $(OBJ_FILES)
	nvcc $(FLAGS) $(OBJ_FILES) $(EXTLIB) -o $(BLDDIR)/$(BUILDNAME)

# compiles src files into object files	
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	nvcc -x cu $(FLAGS) -I. -dc $< -o $@
	
ext:
	cd extlib/jsmn && $(MAKE) && $(MAKE) test
	cd extlib/gmsh_io && $(MAKE) all
		
world:
	mkdir -p $(WORLD)
		
clean:
	rm -r $(OBJDIR)/*
	rm -f $(BLDDIR)/$(BUILDNAME)
	cd extlib/jsmn && $(MAKE) clean
	cd extlib/gmsh_io && $(MAKE) clean

clear vtk: $(VTKDIR)
	rm -f $(VTKDIR)/*

clear doc: $(DOCDIR)
	rm -f -r $(DOCDIR)/*
	
