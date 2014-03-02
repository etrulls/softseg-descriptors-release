fprintf('1 of 2: Compiling SID binaries\n');
cd sid
makefile
fprintf('2 of 2: Compiling SIFT-flow\n');
cd ../sift-flow/mex
mex mexDiscreteFlow.cpp BPFlow.cpp Stochastic.cpp
cd ../..
fprintf('Done!\n');