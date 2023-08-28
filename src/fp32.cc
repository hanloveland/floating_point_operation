#include "fp32.h"
#include <iostream>


#define K_WIDTH 32
#define E_WIDTH 8
#define M_WIDTH 23
#define E_MASK 0xFF
#define M_MASK 0x7FFFFF
#define E_BIAS 127 
#define E_MAX 255
#define E_MIN 0
#define qNaN 0xFFFFFFFF //qNaN
#define pInf 0x7F800000
#define nInf 0xFF800000
#define pzero 0x00000000
#define nzero 0x80000000

// Use RoundTiesToEven Rounding

FP32::FP32(fp32_u uvalue) : value(uvalue) {}
FP32::FP32(float fvalue) : value(fp32_u(fvalue)) {}
FP32::FP32(uint32_t ivalue) : value(fp32_u(ivalue)) {}

FP32 FP32::operator+(const FP32& other) const {
    /*
    if(is_nan() || other.is_nan()) { // return NaN
        std::cout<<"return qNaN"<<std::endl;
        return FP32(uint32_t(qNaN));
    } else if((is_inf() && other.is_inf()) && (get_sign() != other.get_sign())) { // Invalid Operation
        std::cout<<"return qNaN"<<std::endl;
        return FP32(uint32_t(qNaN));     
    } else if(is_inf() || other.is_inf()) { // Infinity Arthmetic
        if((is_inf() && get_sign() == 0) || (other.is_inf() && other.get_sign() == 0)) return FP32(uint32_t(pInf));  
        else                                                                           return FP32(uint32_t(nInf));  
    } else if(is_subnor() || other.is_subnor()) { // Normal Operation 

    } else { // Normal Operation
        std::cout<<"Normal Addition"<<std::endl;        
        std::cout<<" Sign :"<<get_sign()<<" / Sign :"<<other.get_sign()<<std::endl;
        std::cout<<" Expo :"<<get_expo()<<" / Expo :"<<other.get_expo()<<std::endl;
        std::cout<<" Mantissa :"<<get_emani()<<" / Mantissa :"<<other.get_emani()<<std::endl;

        uint32_t pre_min_e = get_expo() >= other.get_expo() ? other.get_expo() : get_expo(); // choose mininum exponent
        uint32_t pre_diff_a_e = get_expo() - pre_min_e;
        uint32_t pre_diff_b_e = other.get_expo() - pre_min_e;

        uint32_t nor_m_a = (uint64_t)get_emani() << pre_diff_a_e;
        uint32_t nor_m_b = (uint64_t)other.get_emani() << pre_diff_a_e;

        uint32_t pre_diff_b_e = other.get_expo() - pre_min_e;

        uint32_t pre_r_e = get_expo() + other.get_expo();   // Exponent (Shifted by the bias, Need to compensate in normalization stage)
        uint64_t pre_r_m = (uint64_t)get_emani() * (uint64_t)other.get_emani(); // Multiply Mantissas        
    }
    */
}
FP32 FP32::operator-(const FP32& other) const {
    //TBD   
}

FP32 FP32::operator*(const FP32& other) const {
    // classify op1/op2 into normal, inf, NaN
    if(is_nan() || other.is_nan()) { // return NaN
        std::cout<<"return qNaN"<<std::endl;
        return FP32(uint32_t(qNaN));
    } else if((is_inf() && other.is_zero()) || (is_zero() && other.is_inf())) { // return NaN
        std::cout<<"return qNaN"<<std::endl;
        return FP32(uint32_t(qNaN));
    } else if(is_inf() || other.is_inf()) {
        std::cout<<"return q/n Inf"<<std::endl;
        if(get_sign() != other.get_sign()) return FP32(uint32_t(nInf));
        else                               return FP32(uint32_t(pInf));    
    } else if(is_subnor() || other.is_subnor()) {
        std::cout<<"return surbnormal (TBD)"<<std::endl;  
        //TBD
    } else {
        std::cout<<"Normal Multiplication"<<std::endl;        
        std::cout<<" Sign :"<<get_sign()<<" / Sign :"<<other.get_sign()<<std::endl;
        std::cout<<" Expo :"<<std::hex<<get_expo()<<" / Expo :"<<other.get_expo()<<std::endl;
        std::cout<<" Mantissa :"<<std::hex<<get_emani()<<" / Mantissa :"<<other.get_emani()<<std::endl;
        uint32_t pre_r_s = get_sign() * other.get_sign();   // Sign 
        uint32_t pre_r_e = get_expo() + other.get_expo();   // Exponent (Shifted by the bias, Need to compensate in normalization stage)
        uint64_t pre_r_m = (uint64_t)get_emani() * (uint64_t)other.get_emani(); // Multiply Mantissas
        uint32_t norm_r_s,norm_r_e,norm_r_m;
        uint32_t norm_grs;
        uint32_t norm_round_up_bits;

        //Normalization 
        norm_r_s = pre_r_s;
        std::cout<<" pre_r_m :"<<std::hex<<pre_r_m<<std::endl;
        if(pre_r_m >> (M_WIDTH + M_WIDTH + 1) == 1) {
            norm_r_e = pre_r_e + 1;
            //pre_r_m >> (M_WIDTH+1-3)
            norm_grs = (pre_r_m >> (M_WIDTH+1-3-20)) & 0xFFFFFF;
            norm_round_up_bits = get_round_bits(norm_grs);
            norm_r_m = (((pre_r_m >> (M_WIDTH+1)) & M_MASK) + norm_round_up_bits);
            std::cout<<" EXP +1 "<<std::endl;
            std::cout<<" norm_grs :"<<std::hex<<norm_grs<<std::endl;
            std::cout<<" norm_r_m :"<<std::hex<<norm_r_m<<std::endl;            
            // Round-up Using RoundTiesToEven
        }
        else {
            norm_r_e = pre_r_e;
            norm_grs = (pre_r_m >> (M_WIDTH-3-20)) & 0xFFFFFF;
            norm_round_up_bits = get_round_bits(norm_grs);
            norm_r_m = (((pre_r_m >> (M_WIDTH)) & M_MASK) + norm_round_up_bits);
            std::cout<<" EXP  "<<std::endl;
            std::cout<<" norm_grs :"<<std::hex<<norm_grs<<std::endl;
            std::cout<<" norm_r_m :"<<std::hex<<norm_r_m<<std::endl;               
        }
        
        if(norm_r_e >= (E_MAX + E_BIAS)) { //Overflow
            if(norm_r_s == 0x1) return FP32(uint32_t(nInf));
            else                return FP32(uint32_t(pInf));
        } else if(norm_r_e == E_BIAS) { // Subnormal (maybe not support?)
           // TBD
        } else if(norm_r_e < E_BIAS) { //Underflow 
            if(norm_r_s == 0x1) {
                return FP32(uint32_t(nzero));
            }
            else {
                return FP32(uint32_t(pzero));
            }               
        } else { // Normal Number Operation
            uint32_t norm_r_e_com = norm_r_e - E_BIAS;
            uint32_t final_r = norm_r_s << (E_WIDTH + M_WIDTH) | norm_r_e_com << M_WIDTH | norm_r_m;
            return FP32(final_r);
        }   
    }
}

FP32 FP32::operator/(const FP32& other) const {
    //TBD
}

bool FP32::is_inf() const {
    uint32_t fp32_e = get_expo();
    uint32_t fp32_m = get_mani();

    if(fp32_e == E_MASK && fp32_m == 0) return true;
    else                                return false;
}

bool FP32::is_nan() const {
    uint32_t fp32_e = get_expo();
    uint32_t fp32_m = get_mani();

    if(fp32_e == E_MASK && fp32_m != 0) return true;
    else                                return false;
}

bool FP32::is_subnor() const {
    uint32_t fp32_e = get_expo();
    uint32_t fp32_m = get_mani();

    if(fp32_e == 0 && fp32_m != 0) return true;
    else                           return false;    
}

bool FP32::is_zero() const {
    uint32_t fp32_e = get_expo();
    uint32_t fp32_m = get_mani();

    if(fp32_e == 0 && fp32_m == 0) return true;
    else                           return false;    
}

uint32_t FP32::get_sign() const {
    uint32_t fp32_s = (value.ival >> (M_WIDTH + E_WIDTH)) & 0x1;
    return fp32_s;
}

uint32_t FP32::get_expo() const {
    uint32_t fp32_e = (value.ival >> M_WIDTH) & E_MASK;
    return fp32_e;
}

uint32_t FP32::get_mani() const {
    uint32_t fp32_m = value.ival & M_MASK;
    return fp32_m;
}

uint32_t FP32::get_emani() const {
    uint32_t fp32_m;
    if(is_subnor()) fp32_m = value.ival & M_MASK;
    else            fp32_m = (0x1 << M_WIDTH) | (value.ival & M_MASK);
    return fp32_m;
}

uint32_t FP32::get_round_bits(uint32_t norm_grs) const {
    // 00xxxxxxx.. -> round_down
    // 010000...00 -> round_down
    // 10xxx...xxx -> round_down
    // other.. round_up

    uint32_t norm_round_up_bits;
    if(norm_grs <= 0x400000) norm_round_up_bits = 0;
    else if(norm_grs>>22 == 0x2) norm_round_up_bits = 0;
    else norm_round_up_bits = 1;
    return norm_round_up_bits;
}

