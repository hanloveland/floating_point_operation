#include <iostream>
#include <iomanip> // for std::setprecision()
#include "fp32.h"
#include <random>

int main() {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 10);

    uint32_t pass = 0;
    uint32_t fail = 0;

    for(int i=0;i<1;i++) {
    float a = dist(e2);
    float b = dist(e2);
    float c = a / b;

    FP32 fp_a = FP32(a);
    FP32 fp_b = FP32(b);
    FP32 fp_exp_c = FP32(a/b);
    FP32 fp_c = fp_a / fp_b;
    std::cout<<std::setprecision(16);
    std::cout<<"a :"<<fp_a.value.fval<<std::hex<<" -> 0x"<<fp_a.value.ival<<std::endl;
    std::cout<<"b :"<<fp_b.value.fval<<std::hex<<" -> 0x"<<fp_b.value.ival<<std::endl;

    std::cout<<"Exp"<<std::endl;
    std::cout<<" --> a / b: "<<fp_exp_c.value.fval<<std::endl;
    std::cout<<" --> Sign "<<std::hex<<fp_exp_c.get_sign()<<std::endl;
    std::cout<<" --> Exponent "<<std::hex<<fp_exp_c.get_expo()<<std::endl;
    std::cout<<" --> Mantissa "<<std::hex<<fp_exp_c.get_mani()<<std::endl;   

    std::cout<<"Calc"<<std::endl;
    std::cout<<" --> a / b: "<<fp_c.value.fval<<std::endl;
    std::cout<<" --> Sign "<<std::hex<<fp_c.get_sign()<<std::endl;
    std::cout<<" --> Exponent "<<std::hex<<fp_c.get_expo()<<std::endl;
    std::cout<<" --> Mantissa "<<std::hex<<fp_c.get_mani()<<std::endl;   

    if(fp_exp_c.value.ival == fp_c.value.ival) {
        pass++;
    } else {
        fail++;
        break;
    }
    }

    std::cout<<"Pass :"<<std::dec<<pass<<" / Fail :"<<fail<<std::endl;
    //std::cout<<"Hex : 0x"<<std::hex<<a.ival<<std::endl;
    //std::cout<<" ---> S: "<<a_S<<std::dec<<"("<<a_S<<")"<<std::endl;
    //std::cout<<" ---> E: "<<a_E<<std::dec<<"("<<a_E<<")"<<std::endl;
    //std::cout<<" ---> T: "<<a_T<<std::dec<<"("<<a_T<<")"<<std::endl;     
    //
    /*
    uint32_t fp32_k = 32;
    uint32_t fp32_w = 8;
    uint32_t fp32_t = 23;
    uint32_t fp32_e_mask = 0xFF;
    uint32_t fp32_t_mask = 0x7FFFFF;
    uint32_t fp32_bias = 127;

    uint32_t is_NaN;
    uint32_t is_pInf;
    uint32_t is_nInf;

    float f = 3.2;
    float f_b = 6.3;
    float f_c_exp = f + f_b;
    fp32 a = fp32(f);
    fp32 b = fp32(f_b);
    fp32 c_exp = fp32(f_c_exp);

    uint32_t a_S = (a.ival >> (fp32_w+fp32_t)) & 0x1; 
    uint32_t a_E = (a.ival >> fp32_t) & fp32_e_mask;
    uint32_t a_T = a.ival & fp32_t_mask;

    uint32_t b_S = (b.ival >> (fp32_w+fp32_t)) & 0x1; 
    uint32_t b_E = (b.ival >> fp32_t) & fp32_e_mask;
    uint32_t b_T = b.ival & fp32_t_mask;

    uint32_t c_exp_S = (c_exp.ival >> (fp32_w+fp32_t)) & 0x1; 
    uint32_t c_exp_E = (c_exp.ival >> fp32_t) & fp32_e_mask;
    uint32_t c_exp_T = c_exp.ival & fp32_t_mask;

    std::cout<<"Floating Point: "<<f<<std::endl;
    std::cout<<"Hex : 0x"<<std::hex<<a.ival<<std::endl;
    std::cout<<" ---> S: "<<a_S<<std::dec<<"("<<a_S<<")"<<std::endl;
    std::cout<<" ---> E: "<<a_E<<std::dec<<"("<<a_E<<")"<<std::endl;
    std::cout<<" ---> T: "<<a_T<<std::dec<<"("<<a_T<<")"<<std::endl; 

    std::cout<<"Floating Point: "<<f_b<<std::endl;
    std::cout<<"Hex : 0x"<<std::hex<<b.ival<<std::endl;
    std::cout<<" ---> S: "<<b_S<<std::dec<<"("<<b_S<<")"<<std::endl;
    std::cout<<" ---> E: "<<b_E<<std::dec<<"("<<b_E<<")"<<std::endl;
    std::cout<<" ---> T: "<<b_T<<std::dec<<"("<<b_T<<")"<<std::endl; 

    std::cout<<"Floating Point: "<<f_c_exp<<std::endl;
    std::cout<<"Hex : 0x"<<std::hex<<c_exp.ival<<std::endl;
    std::cout<<" ---> S: "<<c_exp_S<<std::dec<<"("<<c_exp_S<<")"<<std::endl;
    std::cout<<" ---> E: "<<c_exp_E<<std::dec<<"("<<c_exp_E<<")"<<std::endl;
    std::cout<<" ---> T: "<<c_exp_T<<std::dec<<"("<<c_exp_T<<")"<<std::endl;    
    */
    #if 0
    uint32_t c_S = a_S * b_S; // Signed Mul. 
    uint32_t c_E = a_E + b_E - fp32_bias; // Add Exponent 
    // Significand Mul. 
    uint64_t c_T = (((uint64_t)(a_T|1<<fp32_t) * (uint64_t)(b_T|1<<fp32_t)) >> (fp32_t));
    // Normalize
    uint32_t nor_bits = 0;
    uint64_t nor_c_T = c_T >> fp32_t+1;
    while(nor_c_T != 0) {
        nor_bits++;
        nor_c_T = nor_c_T >> 1;
    }
    nor_c_T = (c_T >> nor_bits) & 0x3FFFFF;
    
    c_T = nor_c_T;
    c_E+=nor_bits;
    uint32_t c_int = c_S << (fp32_w+fp32_t) | c_E<< (fp32_t) | (uint32_t)c_T;
    fp32 c = fp32(c_int);

    std::cout<<"Floating Point: "<<(a.fval*b.fval)<<std::endl;

    std::cout<<"Floating Point: "<<(c.fval)<<std::endl;
    std::cout<<"Hex : 0x"<<std::hex<<c.ival<<std::endl;
    std::cout<<" ---> S: "<<c_S<<std::dec<<"("<<c_S<<")"<<std::endl;
    std::cout<<" ---> E: "<<c_E<<std::dec<<"("<<c_E<<")"<<std::endl;
    std::cout<<" ---> T: "<<c_T<<std::dec<<"("<<c_T<<")"<<std::endl; 

    #endif
    #if 0
    bool large_a = false;
    bool equal_ab = false;
    if(a_E > b_E) large_a = true;
    else if(a_E == b_E) equal_ab = true;
    else large_a = false;

    if(equal_ab) {
        if(a_T > b_T) large_a = true;
        else large_a = false;
    }
    std::cout<<" A is Large?"<<large_a<<std::endl;
    std::cout<<" Equal ? "<<equal_ab<<std::endl;


    uint32_t diff_exp;
    if(large_a) diff_exp = a_E - b_E;
    else        diff_exp = b_E - a_E;
    std::cout<<" diff_exp ? "<<diff_exp<<std::endl;


    uint32_t c_E; 
    if(large_a) c_E = a_E; 
    else        c_E = b_E;

    // Significand Mul. 
    uint64_t c_T;
    if(large_a) c_T = (((uint64_t)(a_T|1<<fp32_t) + (uint64_t)(b_T|1<<fp32_t)>>diff_exp));
    else        c_T = ((((uint64_t)(a_T|1<<fp32_t))>>diff_exp + (uint64_t)(b_T|1<<fp32_t)));
    std::cout<<"a_T ? "<<a_T<<std::endl;
    std::cout<<"(uint64_t)(a_T|1<<fp32_t) ? "<<(uint64_t)(a_T|1<<fp32_t)<<std::endl;
    std::cout<<"(uint64_t)(a_T|1<<fp32_t)>>diff_exp  ? "<<((uint64_t)(a_T|1<<fp32_t)>>diff_exp)<<std::endl;
    std::cout<<"b_T ? "<<b_T<<std::endl;
    std::cout<<"(uint64_t)(b_T|1<<fp32_t) ? "<<(uint64_t)(b_T|1<<fp32_t)<<std::endl;
    std::cout<<"(uint64_t)(b_T|1<<fp32_t)>>diff_exp  ? "<<((uint64_t)(b_T|1<<fp32_t)>>diff_exp)<<std::endl;

    uint64_t c_a_ = ((uint64_t)(a_T|1<<fp32_t)>>diff_exp);
    uint64_t c_b_ = (uint64_t)(b_T|1<<fp32_t);
    c_T = c_a_ + c_b_;
    std::cout<<"c_T ? "<<(c_T)<<std::endl;
    uint32_t nor_bits = 0;
    uint64_t nor_c_T = c_T >> fp32_t+1;
    while(nor_c_T != 0) {
        nor_bits++;
        nor_c_T = nor_c_T >> 1;
    }
    nor_c_T = (c_T >> nor_bits) & 0x3FFFFF;
    
    c_T = nor_c_T;
    c_E+=nor_bits;
    uint64_t c_S = 0;
    uint32_t c_int = c_S << (fp32_w+fp32_t) | c_E<< (fp32_t) | (uint32_t)c_T;
    fp32 c = fp32(c_int);

    std::cout<<"Floating Point: "<<(a.fval*b.fval)<<std::endl;

    std::cout<<"Floating Point: "<<(c.fval)<<std::endl;
    std::cout<<"Hex : 0x"<<std::hex<<c.ival<<std::endl;
    std::cout<<" ---> S: "<<c_S<<std::dec<<"("<<c_S<<")"<<std::endl;
    std::cout<<" ---> E: "<<c_E<<std::dec<<"("<<c_E<<")"<<std::endl;
    std::cout<<" ---> T: "<<c_T<<std::dec<<"("<<c_T<<")"<<std::endl; 
    #endif

    return 0;
}