# Builds simflight: the existing sims in simulation/src (closedLoop, openLoop,
# semiOpenLoop, computeControl, computeTATD) run on the generator's params,
# path and Riccati solution, i.e. exactly what the drones flew.
#
# The sources are linked into build/ so their #include "params.hpp" finds the
# generator's params, not the old ones in simulation/src.
#
#     make -C simflight        (run from formationControl/)

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -I/usr/include/eigen3 -I../generator/include

SIM_DIR  = ../../../../simulation/src
GEN_DIR  = ../generator/src

SIM_CPP  = closedLoop.cpp openLoop.cpp semiOpenLoop.cpp computeControl.cpp computeTATD.cpp odes.cpp
SIM_HPP  = simulate.hpp computeControl.hpp computeTATD.hpp
GEN_CPP  = params.cpp pathManager.cpp riccatiSolver.cpp

OBJ = $(addprefix build/, simflight.o $(SIM_CPP:.cpp=.o) $(GEN_CPP:.cpp=.o))

simflight: $(OBJ)
	$(CXX) $(OBJ) -o $@

build/.links:
	mkdir -p build
	for f in $(SIM_CPP) $(SIM_HPP); do ln -sf ../$(SIM_DIR)/$$f build/$$f; done
	for f in $(GEN_CPP); do ln -sf ../$(GEN_DIR)/$$f build/$$f; done
	ln -sf ../simflight.cpp build/simflight.cpp
	touch $@

build/%.o: | build/.links
	$(CXX) $(CXXFLAGS) -MMD -MP -c build/$*.cpp -o $@

clean:
	rm -rf build simflight

-include $(wildcard build/*.d)
