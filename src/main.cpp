#include <iostream>
#include <iomanip> // for std::setprecision()
#include "fp32.h"
#include <random>
#include "FP16G.h"
#include "fp16.h"

int main() {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(-1,1);
    std::uniform_real_distribution<> dist1(-1,1);
    std::uniform_real_distribution<> dist2(-1,1);

    uint64_t iter = 100000000;
/*
    FP16 Max Value: 65504
    mallest positive subnormal number: 0.000000059604645
    largest subnormal number: 0.000060975552
    smallest positive normal number: 0.00006103515625
    largest normal number: 65504

    fp16i bb(0);
    Case
     -> pINF + pINF
     -> pINF + nINF
     -> nINF + nINF
     -> Normal + Normal 
     -> Normal + Subnormal
     -> Subnormal + Subnormal 
*/
    std::cout<<" Iteration : "<<iter<<std::endl;
    for(int j=0;j<9;j++) {
        if(j==0) {
            std::cout<<" Case: pINF + pINF "<<std::endl;
            std::uniform_real_distribution<> dist_1a(65505,65504*4);
            std::uniform_real_distribution<> dist_1b(65505,65504*4);
            dist1 = dist_1a;
            dist2 = dist_1b;
        } else if(j == 1) {
            std::cout<<" Case: pINF + nINF "<<std::endl;
            std::uniform_real_distribution<> dist_1a(65505,65504*4);
            std::uniform_real_distribution<> dist_1b(-65505.0*4,-65505.0);
            dist1 = dist_1a;
            dist2 = dist_1b;            
        } else if(j == 3) {
            std::cout<<" Case: nINF + nINF "<<std::endl;
            std::uniform_real_distribution<> dist_1a(-65505.0*4,-65505.0);
            std::uniform_real_distribution<> dist_1b(-65505.0*4,-65505.0);
            dist1 = dist_1a;
            dist2 = dist_1b;            
        } else if(j == 4) {
            std::cout<<" Case: Normal (-65505,65505) + Normal (-65505,65505)"<<std::endl;
            std::uniform_real_distribution<> dist_1a(-65505,65505);
            std::uniform_real_distribution<> dist_1b(-65505,65505);
            dist1 = dist_1a;
            dist2 = dist_1b;            
        } else if(j == 5) {
            std::cout<<" Case: Normal (-6550,6550) + Normal (-6550,6550)"<<std::endl;
            std::uniform_real_distribution<> dist_1a(-6550,6550);
            std::uniform_real_distribution<> dist_1b(-6550,6550);
            dist1 = dist_1a;
            dist2 = dist_1b;            
        } else if(j == 6) {
            std::cout<<" Case: Normal (-655,655) + Normal (-655,655)"<<std::endl;
            std::uniform_real_distribution<> dist_1a(-655,655);
            std::uniform_real_distribution<> dist_1b(-655,655);
            dist1 = dist_1a;
            dist2 = dist_1b;            
        } else if(j == 7) {
            std::cout<<" Case: Normal (-65,65) + Normal (-65,65)"<<std::endl;
            std::uniform_real_distribution<> dist_1a(-65,65);
            std::uniform_real_distribution<> dist_1b(-65,65);
            dist1 = dist_1a;
            dist2 = dist_1b;            
        } else if(j == 7) {
            std::cout<<" Case: Normal (-6,6) + Normal (-6,6)"<<std::endl;
            std::uniform_real_distribution<> dist_1a(-6,6);
            std::uniform_real_distribution<> dist_1b(-6,6);
            dist1 = dist_1a;
            dist2 = dist_1b;
        } else if(j == 8) {
            std::cout<<" Case: Normal (..,..) + SubNormal"<<std::endl;
            std::uniform_real_distribution<> dist_1a(-0.0060975552,0.0060975552);
            std::uniform_real_distribution<> dist_1b(-0.000060975552,0.000060975552);
            dist1 = dist_1a;
            dist2 = dist_1b;            
        } else if(j == 8) {
            std::cout<<" Case: SubNormal + SubNormal"<<std::endl;
            std::uniform_real_distribution<> dist_1a(-0.000060975552,0.000060975552);
            std::uniform_real_distribution<> dist_1b(-0.000060975552,0.000060975552);
            dist1 = dist_1a;
            dist2 = dist_1b;            
        }        
    for(uint64_t i=0;i<iter;i++) {
        float fp32_a = dist1(e2);
        float fp32_b = dist2(e2);
        float fp32_c = fp32_a + fp32_b;

        FP16g fp16g_a(fp16(0));
        FP16g fp16g_b(fp16(0));
        fp16g_a = fp16(fp32_a);
        fp16g_b = fp16(fp32_b);
        FP16g fp16g_c(fp16g_a + fp16g_b);
        FP16 cc(float(3.4));

        FP16 fp16_a(uint16_t(0));
        FP16 fp16_b(uint16_t(0));
        fp16_a = fp16_u(fp32_a);
        fp16_b = fp16_u(fp32_b);
        FP16 fp16_c(fp16_a + fp16_b);   

        fp16 diff = (fp16g_c.value - fp16_c.value.fval);
        fp16 diff_ratio = diff/fp16g_c.value;
        bool is_fail = false;
        if(fp16g_c.value == 0.0) {
            if(diff != 0) {
                is_fail = true;   
            }
        }
        else if(diff_ratio > 0.001) {
                is_fail = true;
        }
        if(is_fail) {
            std::cout<<" FP32 "<<std::endl;
            std::cout<<std::setprecision(16);
            std::cout<<"  - A   :"<<fp32_a<<std::endl;
            std::cout<<"  - B   :"<<fp32_b<<std::endl;
            std::cout<<"  - A+B :"<<fp32_c<<std::endl;

            std::cout<<" FP16 Golden "<<std::endl;
            std::cout<<std::setprecision(16);
            std::cout<<"  - A   :"<<fp16g_a.value<<" / 0x"<<std::hex<<fp16i(fp16g_a.value).ival<<std::endl;
            std::cout<<"  - B   :"<<fp16g_b.value<<" / 0x"<<std::hex<<fp16i(fp16g_b.value).ival<<std::endl;
            std::cout<<"  - A+B :"<<fp16g_c.value<<" / 0x"<<std::hex<<fp16i(fp16g_c.value).ival<<std::endl;    

            std::cout<<" FP16 Cal "<<std::endl;
            std::cout<<std::setprecision(16);
            std::cout<<"  - A   :"<<fp16_a.value.fval<<" / 0x"<<std::hex<<fp16_a.value.ival<<std::endl;
            std::cout<<"  - B   :"<<fp16_b.value.fval<<" / 0x"<<std::hex<<fp16_b.value.ival<<std::endl;
            std::cout<<"  - A+B :"<<fp16_c.value.fval<<" / 0x"<<std::hex<<fp16_c.value.ival<<std::endl;

            std::cout<<"Fail!"<<std::endl;
            std::cout<<"Diff "<<diff<<std::endl;
            std::cout<<"Dif Ratio "<<diff_ratio<<std::endl;             
            break;
        }
    }
    }
    // fp16 a(3.2);
    // std::cout<<" A :"<<aa.toFloat()<<std::endl;
    // aa.fromFloat(3.2);
    // std::cout<<" A :"<<aa.value<<std::endl;
    // std::cout<<" A :"<<a<<std::endl;
    // ->fromFloat(3.2);
    // aa.fromFloat(3.2);

    return 0;

    uint32_t pass[4] = {0,0,0,0};
    uint32_t fail[4] = {0,0,0,0};
    for(int op=0;op<4;op++) {
        for(int i=0;i<100000;i++) {
            float a = dist(e2);
            float b = dist(e2);

            FP32 fp_a = FP32(a);
            FP32 fp_b = FP32(b);
            FP32 fp_exp_c = FP32(float(0.0));
            if(op == 0)       fp_exp_c = FP32(a+b);
            else if (op == 1) fp_exp_c = FP32(a-b);
            else if (op == 2) fp_exp_c = FP32(a*b);
            else if (op == 3) fp_exp_c = FP32(a/b);
            FP32 fp_c = FP32(float(0.0));
            if(op == 0)       fp_c = fp_a + fp_b;
            else if (op == 1) fp_c = fp_a - fp_b;
            else if (op == 2) fp_c = fp_a * fp_b;
            else if (op == 3) fp_c = fp_a / fp_b;    

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
                pass[op]++;
            } else {
                fail[op]++;
            }
        }
    }
    std::cout<<"Operator[+] Pass :"<<std::dec<<pass[0]<<" / Fail :"<<fail[0]<<std::endl;
    std::cout<<"Operator[-] Pass :"<<std::dec<<pass[1]<<" / Fail :"<<fail[1]<<std::endl;
    std::cout<<"Operator[*] Pass :"<<std::dec<<pass[2]<<" / Fail :"<<fail[2]<<std::endl;
    std::cout<<"Operator[/] Pass :"<<std::dec<<pass[3]<<" / Fail :"<<fail[3]<<std::endl;

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

    return 0;
}