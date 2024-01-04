# floating_point_operation

build:
$ g++ main.cpp fp32.cc FP16G.cc fp16.cc -o main

Running:
$ ./main

C-based bit-level FP16/FP32 operator for designing and verifying a 16bits/32bits floating pointer hardware operator.
1. FP32 Addition
2. FP32 Subtraction
3. FP32 Multiplication
4. FP32 division (occur bit error at last bit of mantisa due to round-up)
5. FP16 Addition/Subtraction (support normal and subnormal) 
6. FP16 Multiplication (support normal and subnormal) 
TODO list
1. add FP16 operator (Division)