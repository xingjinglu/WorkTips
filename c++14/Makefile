CXX=clang++
CXXFLAGS="-v"

all: template14 

template14 : template14.o
	$(CXX) $< -o $@
%.o : %.cc
	$(CXX) -c $(CXXFLAGS) $< -o $@



clean:
	rm $(all) *.o

