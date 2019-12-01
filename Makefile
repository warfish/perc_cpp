CXX = clang++
CXXFLAGS = -O2 -ggdb3 -fno-omit-frame-pointer --std=c++14 -Wall

all: perc

perc: perc.cpp
	$(CXX) $(CXXFLAGS) perc.cpp -o perc

clean:
	rm -f perc
