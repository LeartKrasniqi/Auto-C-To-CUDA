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

PROJ_DEPS = normalize.lo

translate: $(PROJ_DEPS)
	$(ROSE_BUILD)/libtool --mode=link $(CXX) $(CXXFLAGS) -I$(ROSE_INCLUDE_DIR) $(BOOST_CPPFLAGS) -o translate.out $(PROJ_DEPS) translate.cpp $(ROSE_LIBS) 

normalize.lo: ./include/normalize/normalize.cpp ./include/normalize/normalize.hpp 
	$(ROSE_BUILD)/libtool --mode=compile $(CXX) $(CXXFLAGS) -I$(ROSE_INCLUDE_DIR) $(BOOST_CPPFLAGS) -c -o normalize.lo ./include/normalize/normalize.cpp $(ROSE_LIBS) 
clean:
	rm *.out
