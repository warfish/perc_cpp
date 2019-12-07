CXX = clang++
CXXFLAGS = -O0 -ggdb3 -fno-omit-frame-pointer --std=c++14 -Wall -march=native

all: perc

perc: perc.cpp
	$(CXX) $(CXXFLAGS) perc.cpp -o perc

clean:
	rm -f perc
