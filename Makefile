# Location of include directory after "make install"
ROSE_INCLUDE_DIR = $(ROSE_INSTALL)/include/rose

# Location of Boost include directory
BOOST_CPPFLAGS = -pthread -I/usr/include

# Location of lib directory after "make install"
ROSE_LIB_DIR = $(ROSE_INSTALL)/lib

CC = gcc
CXX = g++
CXXFLAGS = -g -O2 -Wall

ROSE_LIBS = $(ROSE_LIB_DIR)/librose.la

PROJ_DEPS = normalize.lo affine.lo dependency.lo parallel.lo kernel.lo preprocess.lo

translate: $(PROJ_DEPS) ./include/loop_attr.hpp
	$(ROSE_BUILD)/libtool --mode=link $(CXX) $(CXXFLAGS) -I$(ROSE_INCLUDE_DIR) $(BOOST_CPPFLAGS) -o translate.out $(PROJ_DEPS) translate.cpp $(ROSE_LIBS) 

normalize.lo: ./include/normalize/normalize.cpp ./include/normalize/normalize.hpp 
	$(ROSE_BUILD)/libtool --mode=compile $(CXX) $(CXXFLAGS) -I$(ROSE_INCLUDE_DIR) $(BOOST_CPPFLAGS) -c -o normalize.lo ./include/normalize/normalize.cpp $(ROSE_LIBS) 

affine.lo: ./include/affine/affine.cpp ./include/affine/affine.hpp ./include/loop_attr.hpp 
	$(ROSE_BUILD)/libtool --mode=compile $(CXX) $(CXXFLAGS) -I$(ROSE_INCLUDE_DIR) $(BOOST_CPPFLAGS) -c -o affine.lo ./include/affine/affine.cpp $(ROSE_LIBS) 

dependency.lo: ./include/dependency/dependency.cpp ./include/dependency/dependency.hpp ./include/loop_attr.hpp
	$(ROSE_BUILD)/libtool --mode=compile $(CXX) $(CXXFLAGS) -I$(ROSE_INCLUDE_DIR) $(BOOST_CPPFLAGS) -c -o dependency.lo ./include/dependency/dependency.cpp $(ROSE_LIBS)

parallel.lo: ./include/parallel/parallel.cpp ./include/parallel/parallel.hpp ./include/loop_attr.hpp dependency.lo kernel.lo 
	$(ROSE_BUILD)/libtool --mode=compile $(CXX) $(CXXFLAGS) -I$(ROSE_INCLUDE_DIR) $(BOOST_CPPFLAGS) -c -o parallel.lo ./include/parallel/parallel.cpp $(ROSE_LIBS) 

kernel.lo: ./include/kernel/kernel.cpp ./include/kernel/kernel.hpp ./include/loop_attr.hpp
	$(ROSE_BUILD)/libtool --mode=compile $(CXX) $(CXXFLAGS) -I$(ROSE_INCLUDE_DIR) $(BOOST_CPPFLAGS) -c -o kernel.lo ./include/kernel/kernel.cpp $(ROSE_LIBS)

preprocess.lo: ./include/preprocess/preprocess.cpp ./include/preprocess/preprocess.hpp 
	$(ROSE_BUILD)/libtool --mode=compile $(CXX) $(CXXFLAGS) -I$(ROSE_INCLUDE_DIR) $(BOOST_CPPFLAGS) -c -o preprocess.lo ./include/preprocess/preprocess.cpp $(ROSE_LIBS)

clean:
	rm *.out *.lo
