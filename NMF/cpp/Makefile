CXX = g++
CFLAGS = $(shell gsl-config --cflags)
CFLAGS_DEBUG = $(shell gsl-config --cflags) -g
LIBS = $(shell gsl-config --libs)

output: segment_NMF.cpp
	$(CXX) $(CFLAGS) segment_NMF.cpp $(LIBS) -o segment_NMF

debug: segment_NMF.cpp
	$(CXX) $(CFLAGS_DEBUG) segment_NMF.cpp $(LIBS) -o segment_NMF_debug

clean:
	rm segment_NMF segment_NMF_debug
