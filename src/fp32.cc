#include "fp32.h"
#include <iostream>

// #define PRINT_DEBUG 1
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
#define DIFF_THR 23
// Use RoundTiesToEven Rounding

FP32::FP32(fp32_u uvalue) : value(uvalue) {}
FP32::FP32(float fvalue) : value(fp32_u(fvalue)) {}
FP32::FP32(uint32_t ivalue) : value(fp32_u(ivalue)) {}

FP32 FP32::operator+(const FP32& other) const {
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
        //TBD (maybe..not support)
        std::cout<<"return surbnormal (TBD)"<<std::endl;  
        return FP32(uint32_t(pzero));     
    } else { // Normal Operation
        std::cout<<"Normal Addition"<<std::endl;        

        uint32_t is_large_a = get_expo() >= other.get_expo() ? 1 : 0;   // Compare Exponent for Input A and Input B
        uint32_t pre_diff  = (is_large_a == 1) ? (get_expo() - other.get_expo()) :
                                                 (other.get_expo() - get_expo());
        if(pre_diff > DIFF_THR) {
            if(is_large_a == 1) return FP32(value.fval);
            else                return other;
        }
        uint32_t post_e = (is_large_a == 1) ? other.get_expo() : get_expo();
        // Normalize mantiassa 
        uint64_t pre_m_a = (is_large_a == 1) ? (uint64_t)get_emani() << pre_diff :
                                               (uint64_t)get_emani();
        uint64_t pre_m_b = (is_large_a == 1) ? (uint64_t)other.get_emani() :
                                               (uint64_t)other.get_emani() << pre_diff;                                            

        //Calcuation (the difference exponents for Input A and B is less then ..24???)
        uint64_t post_m;
        uint32_t post_s;
        if(get_sign() == other.get_sign()) {
            post_m = pre_m_a + pre_m_b; 
            post_s = get_sign();
        } else {
            if(pre_m_a >= pre_m_b) {
                post_m = pre_m_a - pre_m_b;
                post_s = get_sign();
            } else {
                post_m = pre_m_b - pre_m_a;
                post_s = other.get_sign();
            }
        }         
        // Normalize and return Result
        uint32_t nor_s,nor_e,nor_m;
        nor_s = post_s;
        // Find position where the msb is 'one'
        uint32_t msb_one_pos;
        for(uint32_t pos=0;pos<63;pos++) {
            if((post_m >> pos & 0x1) == 1) msb_one_pos = pos;
        }

        if(msb_one_pos >= 23) {
            u_int32_t shift_pos = msb_one_pos - 23;
            nor_e = post_e + shift_pos;
            nor_m = (post_m >> shift_pos) & M_MASK;
            if(nor_e > E_MAX) {
                if(nor_s == 0) return FP32(uint32_t(pInf));
                else           return FP32(uint32_t(nInf));
            }
            uint32_t nor_grs;
            uint32_t nor_round_up_bits;
            nor_grs = (post_m << (23-shift_pos)) & 0xFFFFFF;
            nor_round_up_bits = get_round_bits(nor_grs);    
            nor_m+=nor_round_up_bits;        
        } else {
            u_int32_t shift_pos = 23-msb_one_pos;
            nor_e = post_e - shift_pos;
            nor_m = (post_m << shift_pos) & M_MASK;     
            if(post_e <= shift_pos) {
                if(nor_s == 0) return FP32(uint32_t(pzero));
                else           return FP32(uint32_t(nzero));                
            }
        }
        uint32_t final_r = nor_s << (E_WIDTH + M_WIDTH) | nor_e << M_WIDTH | nor_m;
        
        #ifdef PRINT_DEBUG
        std::cout<<"Normal Addition"<<std::endl;        
        std::cout<<" Operand A Sign :"<<std::hex<<get_sign()<<" / Sign :"<<other.get_sign()<<std::endl;
        std::cout<<" Expo :"<<std::hex<<get_expo()<<" / Expo :"<<other.get_expo()<<std::endl;
        std::cout<<" Mantissa :"<<std::hex<<get_emani()<<" / Mantissa :"<<other.get_emani()<<std::endl;

        std::cout<<" Pre M for A :"<<std::hex<<pre_m_a<<std::endl;
        std::cout<<" Pre M for B :"<<std::hex<<pre_m_b<<std::endl;

        std::cout<<" Post S  "<<post_s<<std::endl;
        std::cout<<" Post E :"<<std::hex<<post_e<<std::endl;
        std::cout<<" Post M :"<<std::hex<<post_m<<std::endl;   

        std::cout<<" Normalized S  "<<nor_s<<std::endl;
        std::cout<<" Normalized E :"<<std::hex<<nor_e<<std::endl;
        std::cout<<" Normalized M :"<<std::hex<<nor_m<<std::endl;                    
        #endif                       
        return FP32(final_r);   
    }
}
FP32 FP32::operator-(const FP32& other) const {
    FP32 a = FP32(value.fval);
    FP32 b = FP32(other.value.fval);
    uint32_t sign_b;
    if(b.get_sign() == 1) sign_b = 0;
    else                  sign_b = 1;
    b.value.ival = sign_b<<(K_WIDTH-1) | b.get_expo() << (M_WIDTH) | b.get_mani();
    return a+b;
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
        if(get_sign() != other.get_sign()) return FP32(uint32_t(nzero));
        else                               return FP32(uint32_t(pzero));        
    } else {
        std::cout<<"Normal Multiplication"<<std::endl;        
        uint32_t pre_r_s = get_sign() * other.get_sign();   // Sign 
        uint32_t pre_r_e = get_expo() + other.get_expo();   // Exponent (Shifted by the bias, Need to compensate in normalization stage)
        uint64_t pre_r_m = (uint64_t)get_emani() * (uint64_t)other.get_emani(); // Multiply Mantissas
        uint32_t norm_r_s,norm_r_e,norm_r_m;
        uint32_t norm_grs;
        uint32_t norm_round_up_bits;

        //Normalization 
        norm_r_s = pre_r_s;
        if(pre_r_m >> (M_WIDTH + M_WIDTH + 1) == 1) {
            norm_r_e = pre_r_e + 1;
            norm_grs = (pre_r_m >> (M_WIDTH+1-3-20)) & 0xFFFFFF;
            norm_round_up_bits = get_round_bits(norm_grs);
            norm_r_m = (((pre_r_m >> (M_WIDTH+1)) & M_MASK) + norm_round_up_bits);
            // Round-up Using RoundTiesToEven
        }
        else {
            norm_r_e = pre_r_e;
            norm_grs = (pre_r_m >> (M_WIDTH-3-20)) & 0xFFFFFF;
            norm_round_up_bits = get_round_bits(norm_grs);
            norm_r_m = (((pre_r_m >> (M_WIDTH)) & M_MASK) + norm_round_up_bits);
        }
        
        if(norm_r_e >= (E_MAX + E_BIAS)) { //Overflow
            if(norm_r_s == 0x1) return FP32(uint32_t(nInf));
            else                return FP32(uint32_t(pInf));
        } else if(norm_r_e == E_BIAS) { // Subnormal (maybe not support?)
           // TBD (return zero)
            if(norm_r_s == 0x1) return FP32(uint32_t(nzero));
            else                return FP32(uint32_t(pzero));             
        } else if(norm_r_e < E_BIAS) { //Underflow 
            if(norm_r_s == 0x1) return FP32(uint32_t(nzero));
            else return                FP32(uint32_t(pzero));
        } else { // Normal Number Operation
            uint32_t norm_r_e_com = norm_r_e - E_BIAS;
            uint32_t final_r = norm_r_s << (E_WIDTH + M_WIDTH) | norm_r_e_com << M_WIDTH | norm_r_m;
            #ifdef PRINT_DEBUG
                std::cout<<"Normal Multiplication"<<std::endl;        
                std::cout<<" Sign :"<<get_sign()<<" / Sign :"<<other.get_sign()<<std::endl;
                std::cout<<" Expo: "<<std::hex<<get_expo()<<" / Expo: "<<other.get_expo()<<std::endl;
                std::cout<<" Mantissa: "<<std::hex<<get_emani()<<" / Mantissa: "<<other.get_emani()<<std::endl;

                std::cout<<" Pre S:"<<std::hex<<pre_r_s<<std::endl;
                std::cout<<" Pre E:"<<std::hex<<pre_r_e<<std::endl;
                std::cout<<" Pre M:"<<std::hex<<pre_r_m<<std::endl;

                std::cout<<" Post & Normalized S:"<<std::hex<<norm_r_s<<std::endl;
                std::cout<<" Post & Normalized E:"<<std::hex<<norm_r_e_com<<std::endl;
                std::cout<<" Post & Normalized M:"<<std::hex<<norm_r_m<<std::endl;                
            #endif              
            return FP32(final_r);
        }   
    }
}

FP32 FP32::operator/(const FP32& other) const {
    // classify op1/op2 into normal, inf, NaN
    if(is_nan() || other.is_nan()) { // return NaN
        std::cout<<"return qNaN"<<std::endl;
        return FP32(uint32_t(qNaN));
    } else if((is_inf() && other.is_inf()) || (is_zero() && other.is_zero())) { // return NaN
        std::cout<<"return qNaN"<<std::endl;
        return FP32(uint32_t(qNaN));
    } else if(is_inf() && !other.is_inf() && !other.is_zero()) {
        std::cout<<"return Inf"<<std::endl;
        if(get_sign() != other.get_sign()) return FP32(uint32_t(nInf));
        else                               return FP32(uint32_t(pInf));    
    } else if(!is_inf() && other.is_inf()) {
        std::cout<<"return zero"<<std::endl;
        if(get_sign() != other.get_sign()) return FP32(uint32_t(nzero));
        else                               return FP32(uint32_t(pzero)); 
     } else if(!is_inf() && other.is_zero()) {
        std::cout<<"return inf"<<std::endl;
        if(get_sign() != other.get_sign()) return FP32(uint32_t(nInf));
        else                               return FP32(uint32_t(pInf));        
    } else if(is_subnor() || other.is_subnor()) {
        //TBD
        std::cout<<"return surbnormal (TBD)"<<std::endl;  
        if(get_sign() != other.get_sign()) return FP32(uint32_t(nzero));
        else                               return FP32(uint32_t(pzero));         
    } else {
        std::cout<<"Normal Division"<<std::endl;        
        uint32_t pre_s,pre_e;
        uint64_t pre_m;
        if(get_sign() != other.get_sign()) pre_s = 1;
        else                               pre_s = 0;

        uint64_t sh_manti = 16; 
        pre_e = (E_BIAS + get_expo()) - other.get_expo();
        pre_m = ((uint64_t)(get_emani())<<(M_WIDTH+1+sh_manti))/(uint64_t)(other.get_emani()); 

        uint32_t nor_s,nor_e,nor_m;//
        nor_s = pre_s;

        uint32_t norm_grs;
        uint32_t norm_round_up_bits;
        if(pre_m >> (M_WIDTH + 1 + sh_manti) == 1) {
            nor_e = pre_e;
            norm_grs = (pre_m << (M_WIDTH+1-sh_manti-2)) & 0xFFFFFF;
            norm_round_up_bits = get_round_bits(norm_grs);
            nor_m = (((pre_m >> (1+sh_manti)) & M_MASK) + norm_round_up_bits);       
            // Round-up Using RoundTiesToEven
        }
        else {
            nor_e = pre_e - 1;
            norm_grs = (pre_m << (M_WIDTH+1-sh_manti-1)) & 0xFFFFFF;
            norm_round_up_bits = get_round_bits(norm_grs);
            nor_m = (((pre_m >> sh_manti) & M_MASK) + norm_round_up_bits);             
        }
            uint32_t final_r = nor_s << (E_WIDTH + M_WIDTH) | nor_e << M_WIDTH | nor_m;
            #ifdef PRINT_DEBUG
                std::cout<<"Normal Division"<<std::endl;        
                std::cout<<" Sign :"<<get_sign()<<" / Sign :"<<other.get_sign()<<std::endl;
                std::cout<<" Expo: "<<std::hex<<get_expo()<<" / Expo: "<<other.get_expo()<<std::endl;
                std::cout<<" Mantissa: "<<std::hex<<get_emani()<<" / Mantissa: "<<other.get_emani()<<std::endl;

                std::cout<<" Pre S:"<<std::hex<<pre_s<<std::endl;
                std::cout<<" Pre E:"<<std::hex<<pre_e<<std::endl;
                std::cout<<" Pre M:"<<std::hex<<pre_m<<std::endl;

                std::cout<<" Post & Normalized S:"<<std::hex<<nor_s<<std::endl;
                std::cout<<" Post & Normalized E:"<<std::hex<<nor_e<<std::endl;
                std::cout<<" Post & Normalized M:"<<std::hex<<nor_m<<std::endl;                
            #endif
            return FP32(final_r);
    }
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

