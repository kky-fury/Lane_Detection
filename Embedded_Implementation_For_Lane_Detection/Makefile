#Makefile for compiling c++ code with opencv and gpumodule

CC=nvcc
C=g++ 
CFLAGS=`pkg-config --cflags opencv`
LDFLAGS=`pkg-config --libs opencv`
I_PATH=-I/usr/local/cuda/include/
L_PATH=-L/usr/local/cuda/lib64/
Include=-I/home/nvidia/Embedded_Lane_Detection/include/

BIN_1=lane_detect
BIN_2 = hough
BIN_3= bev

OBJECTS = $(BIN_1) $(BIN_2) $(BIN_3)

threshold_2:Hough_Transform.cu
	$(CC) -std=c++11 -arch=sm_62 Hough_Transform.cu -o $(BIN_2) $(Include) $(CFLAGS) $(LDFLAGS) $(I_PATH) $(L_PATH) -lopencv_gpu

threshold_1:Lane_Detection.cpp 
	$(CC) -std=c++11 -arch=sm_62 Lane_Detection.cpp -o $(BIN_1) $(Include) $(CFLAGS) $(LDFLAGS) $(I_PATH) $(L_PATH) -lopencv_gpu
threshold_3:generateBEV.cpp 
	$(C) -std=c++11 generateBEV.cpp -o $(BIN_3) $(Include) $(I_PATH) $(L_PATH)

clean:
	rm $(OBJECTS) 
