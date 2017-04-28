#Makefile for compiling c++ code with opencv and gpumodule

CC=nvcc
C=g++ 
CFLAGS=`pkg-config --cflags opencv`
LDFLAGS=`pkg-config --libs opencv`
I_PATH=-I/usr/local/cuda/include/
L_PATH=-L/usr/local/cuda/lib64/
C_I_PATH=-I/home/nvidia/Embedded_Lane_Detection/include/


#BIN_1=lane_detect
#BIN_2=hough
#BIN_3=bev
#BIN_4=matrix_mul 
BIN_5=bev_thrust
#BIN_6=pre_process



#OBJECTS = $(BIN_1) $(BIN_2) $(BIN_3) $(BIN_4) $(BIN_5) $(BIN_6)

OBJECTS=Lane_Detection_main.o pre_process.o bev_thrust.o rgb2gray.o hough_transform.o line.o line_fitting.o

#threshold_2:Hough_Transform.cu
#	$(CC) -std=c++11 -arch=sm_62 Hough_Transform.cu -o $(BIN_2) $(Include) $(CFLAGS) $(LDFLAGS) $(I_PATH) $(L_PATH) -lopencv_gpu

#threshold_1:Lane_Detection.cpp 
	#$(CC) -std=c++11 -arch=sm_62 Lane_Detection.cpp -o $(BIN_1) $(Include) $(CFLAGS) $(LDFLAGS) $(I_PATH) $(L_PATH) -lopencv_gpu
#threshold_3:generateBEV.cpp 
#	$(C) -std=c++11 generateBEV.cpp -o $(BIN_3) $(Include) $(CFLAGS) $(LDFLAGS) $(I_PATH) $(L_PATH) -fopenmp
#matrix_mul:Mmul_thrust.cu
#	$(CC) -lcublas -lcuda -lcurand -std=c++11 -arch=sm_62 Mmul_thrust.cu -o $(BIN_4) $(Include) $(I_PATH) $(L_PATH)

#bev_thrust:generateBEV_with_thrust.cu
#	$(CC)  -lcuda -O3 -arch=sm_62 -std=c++11 generateBEV_with_thrust.cu -o $(BIN_5) $(Include) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS) 
#pre_process:pre_processing.cu
#	$(CC) -lcuda -O3 -arch=sm_62 -std=c++11 pre_processing.cu -o $(BIN_6) $(Include) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS)

#bev_thrust:./src/generateBEV_with_thrust.cu
#	$(CC) -lcuda -O3 -arch=sm_62 -std=c++11 ./src/generateBEV_with_thrust.cu -o $(BIN_5) $(C_I_PATH) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS) 

lane_detect:$(OBJECTS)
	$(CC) -O3 -arch=sm_62 -std=c++11 $(OBJECTS) -o ./build/lane_detect $(C_I_PATH) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS)

pre_process.o:./src/pre_processing.cu 
	$(CC) -c -O3 -arch=sm_62 -std=c++11 ./src/pre_processing.cu $(C_I_PATH) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS) -o pre_process.o

bev_thrust.o:./src/generate_BEV.cu 
	$(CC) -c -O3 -arch=sm_62 -std=c++11 ./src/generate_BEV.cu $(C_I_PATH) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS) -o bev_thrust.o
 
Lane_Detection_main.o:./src/Lane_Detection_main.cpp 
	$(CC) -c -O3  -arch=sm_62 -std=c++11 ./src/Lane_Detection_main.cpp  $(C_I_PATH) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS) -o Lane_Detection_main.o

rgb2gray.o:./src/rgb2gray.cu 
	$(CC) -c -O3 -arch=sm_62 -std=c++11 ./src/rgb2gray.cu $(C_I_PATH) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS) -o rgb2gray.o

hough_transform.o:./src/Hough_Transform.cu 
	$(CC) -c -O3 -arch=sm_62 -std=c++11 ./src/Hough_Transform.cu $(C_I_PATH) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS) -o hough_transform.o 

line.o:./src/Line.cpp
	$(C) -c -O3 -std=c++11 ./src/Line.cpp $(C_I_PATH) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS) -o line.o 
line_fitting.o:./src/line_fitting.cpp
	$(C) -c -O3 -std=c++11 ./src/line_fitting.cpp $(C_I_PATH) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS) -o line_fitting.o 

clean:
	rm *.o
	rm ./build/lane_detect
