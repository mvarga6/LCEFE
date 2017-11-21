BUILDNAME = gafe6

OBJDIR := obj
SRCDIR := src
VTKDIR := VTKOUT
BLDDIR := builds
WORLD := $(OBJDIR) $(VTKDIR) $(BLDDIR)

EXTLIB = extlib/gmsh_io/libgmsh_io.a extlib/jsmn/libjsmn.a
FLAGS = -lcurand -ccbin=g++ -std=c++11 -Wno-deprecated-gpu-targets

CPP_FILES := $(wildcard $(SRCDIR)/*.cpp)
OBJ_FILES := $(addprefix $(OBJDIR)/,$(notdir $(CPP_FILES:.cpp=.o)))
	
all: world ext $(OBJ_FILES)
	nvcc $(FLAGS) $(OBJ_FILES) $(EXTLIB) -o $(BLDDIR)/$(BUILDNAME)

debug: world ext $(OBJ_FILES)
	nvcc $(FLAGS) -G -g $(OBJ_FILES) $(EXTLIB) -o $(BLDDIR)/$(BUILDNAME)
	
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	nvcc -x cu $(FLAGS) -I. -dc $< -o $@
	
ext:
	cd extlib/jsmn && $(MAKE) && $(MAKE) test
	cd extlib/gmsh_io && $(MAKE) all
		
world:
	mkdir -p $(WORLD)
		
clean:
	rm -r $(OBJDIR)
	rm -f $(BLDDIR)/$(BUILDNAME)
	cd extlib/jsmn && $(MAKE) clean
	cd extlib/gmsh_io && $(MAKE) clean

clear: $(VTKDIR) $(CACHE)
	rm -f $(VTKDIR)/*
	
