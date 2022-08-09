CXX=clang++
#CXXFLAGS=

all: friend_class override2


override2 : override.o
	$(CXX) $< -o $@
friend_class : friend_class.o
	$(CXX) $< -o $@
%.o : %.cc
	$(CXX) -c $(CXXFLAGS) $< -o $@



clean:
	rm $(all) *.o

