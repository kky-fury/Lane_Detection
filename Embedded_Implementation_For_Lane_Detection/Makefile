#Makefile for compiling c++ code with opencv and gpumodule

#CC=nvcc
CC=g++
CFLAGS=`pkg-config --cflags opencv`
LDFLAGS=`pkg-config --libs opencv`
I_PATH=-I/usr/local/cuda/include/
L_PATH=-L/usr/local/cuda/lib64/
Include=-I/home/nvidia/Embedded_Lane_Detection/include/

BIN=lane_detect
threshold:Lane_Detection.cpp 
	$(CC) -std=c++11 Lane_Detection.cpp -o $(BIN) $(Include) $(CFLAGS) $(LDFLAGS) $(I_PATH) $(L_PATH) -lopencv_gpu
