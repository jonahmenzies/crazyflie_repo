CXX      = g++
CXXFLAGS = -std=c++17 -O2 -I/usr/include/eigen3 -Isrc
SRC      = $(wildcard src/*.cpp)
OBJ      = $(SRC:.cpp=.o)
DEP      = $(OBJ:.o=.d)

sim: $(OBJ)
	$(CXX) $(OBJ) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

run: sim
	./sim

plot: sim
	./sim
	python plot.py

clean:
	rm -f $(OBJ) $(DEP) sim

-include $(DEP)
