CXX      = g++
CXXFLAGS = -std=c++17 -O2 -I/usr/include/eigen3 -Iinclude
SRC      = $(wildcard src/*.cpp)
OBJ      = $(SRC:.cpp=.o)
DEP      = $(OBJ:.o=.d)

generate: $(OBJ)
	$(CXX) $(OBJ) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

run: generate
	mkdir -p ../arrays
	./generate

clean:
	rm -f $(OBJ) $(DEP) generate

-include $(DEP)
