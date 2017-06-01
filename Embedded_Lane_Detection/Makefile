#Makefile for compiling c++ code with opencv and gpumodule

CC=nvcc
C=g++ 
CFLAGS=`pkg-config --cflags opencv`
LDFLAGS=`pkg-config --libs opencv`
I_PATH=-I/usr/local/cuda/include/
L_PATH=-L/usr/local/cuda/lib64/
C_I_PATH=-I/home/nvidia/Embedded_Lane_Detection/include/


OBJECTS=Lane_Detection_main.o pre_process.o bev_thrust.o rgb2gray.o hough_transform.o line.o poly.o line_fitting.o readcalib.o

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
#spline.o:./src/spline_fitting.cpp   
#	$(C) -c -O3 -std=c++11 ./src/spline_fitting.cpp $(C_I_PATH) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS) -o spline.o
#spline.o:./src/spline_fitting_with_lib.cpp 
#	$(C) -c -O3 -std=c++11 ./src/spline_fitting_with_lib.cpp $(C_I_PATH) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS) -o spline.o
line_fitting.o:./src/line_fitting.cpp
	$(C) -c -O3 -std=c++11 ./src/line_fitting.cpp $(C_I_PATH) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS) -o line_fitting.o 
poly.o:./src/fit_poly.cpp 
	$(C) -c -O3 -std=c++11 -DBOOST_UBLAS_NDEBUG ./src/fit_poly.cpp  $(C_I_PATH) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS) -o poly.o
readcalib.o:./src/readcalib.cpp 
	$(C) -c -O3 -std=c++11 ./src/readcalib.cpp $(C_I_PATH) $(I_PATH) $(L_PATH) $(CFLAGS) $(LDFLAGS) -o readcalib.o

clean:
	rm *.o
	rm ./build/lane_detect
