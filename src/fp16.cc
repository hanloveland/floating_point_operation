#include "fp16.h"
#include <iostream>

// #define PRINT_DEBUG 1
#define K_WIDTH 16
#define E_WIDTH 5
#define M_WIDTH 10
#define E_MASK 0x1F
#define M_MASK 0x3FF
#define E_BIAS 16
#define E_MAX 31
#define E_MIN 0
#define qNaN 0xFFFF //qNaN
#define pInf 0x7C00
#define nInf 0xFC00
#define pzero 0x0000
#define nzero 0x8000
#define DIFF_THR 11
// Use RoundTiesToEven Rounding

/*
Reference:
[1] https://www.slideserve.com/abbot-casey/floating-point-arithmetic
[2] https://en.wikipedia.org/wiki/Half-precision_floating-point_format
[3] https://www.ecs.umass.edu/ece/koren/arith/slides/
*/



FP16::FP16(fp16_u uvalue) : value(uvalue) {}
FP16::FP16(float fvalue) : value(fp16_u(fvalue)) {}
FP16::FP16(uint16_t ivalue) : value(fp16_u(ivalue)) {}

FP16 FP16::operator+(const FP16& other) const {
    if(is_nan() || other.is_nan()) { // return NaN
        #ifdef PRINT_DEBUG
            std::cout<<"return qNaN"<<std::endl;
        #endif 
        return FP16(uint16_t(qNaN));
    } else if((is_inf() && other.is_inf()) && (get_sign() != other.get_sign())) { // Invalid Operation
        #ifdef PRINT_DEBUG
            std::cout<<"return qNaN"<<std::endl;
        #endif
        return FP16(uint16_t(qNaN));     
    } else if(is_inf() || other.is_inf()) { // Infinity Arthmetic
        if((is_inf() && get_sign() == 0) || (other.is_inf() && other.get_sign() == 0)) return FP16(uint16_t(pInf));  
        else                                                                           return FP16(uint16_t(nInf));  
    // } else if(is_subnor() && other.is_subnor()) { // Sub-normal Operation 
    //     //TBD (maybe..not support)
    //     std::cout<<"return surbnormal (TBD)"<<std::endl;  
    //     return FP16(uint16_t(pzero));     
    } else if(is_zero() || other.is_zero()) { // One of Operands is zero, return the other value
        if(is_zero())   return other;
        else            return FP16(value);
    } else { // (Sub)Normal Operation
        #ifdef PRINT_DEBUG
            std::cout<<"Normal Addition"<<std::endl;      
            std::cout<<" Operand A: "<<std::hex<<value.ival<<" / B: "<<other.value.ival<<std::endl;
            std::cout<<" Operand A Sign: "<<std::hex<<get_sign()<<" / B Sign: "<<other.get_sign()<<std::endl;
            std::cout<<" A Expo: "<<std::hex<<get_expo()<<" / B Expo: "<<other.get_expo()<<std::endl;
            std::cout<<" A Mantissa: "<<std::hex<<get_emani()<<" / B Mantissa: "<<other.get_emani()<<std::endl;
        #endif
        // Compare Exponents & Mantissa 
        uint16_t is_both_sub    = is_subnor() && other.is_subnor();
        uint16_t is_large_a_exp = get_expo() > other.get_expo() ? 1 : 0;        // Compare Exponent for Input A and Input B
        uint16_t is_same_exp    = get_expo() == other.get_expo() ? 1 : 0;       // Compare Exponent for Input A and Input B
        uint16_t is_large_a_man = get_emani() > other.get_emani() ? 1 : 0;
        uint16_t is_large_a     = is_large_a_exp == 1 ? 1 : (is_same_exp == 1 ? (is_large_a_man == 1 ? 1 : 0) : 0);

        uint16_t RawExp     = (is_large_a_exp == 1) ? get_expo() : other.get_expo();
        uint16_t BigMan     = ((is_large_a_exp == 1) || (is_same_exp && is_large_a_man)) ? get_emani() : other.get_emani();
        uint16_t LitMan     = ((is_large_a_exp == 1) || (is_same_exp && is_large_a_man)) ? other.get_emani() : get_emani();

        uint16_t compensated_Exp = (is_subnor() ^ other.is_subnor()) ? 1 : 0;
        uint16_t Exp_diff  = (is_large_a_exp == 1) ? (get_expo() - other.get_expo()) :
                                                     (other.get_expo() - get_expo());
        Exp_diff = Exp_diff - compensated_Exp;
        uint16_t SignOut = is_large_a == 1 ? get_sign() : other.get_sign();
        uint16_t is_Add = (get_sign() == other.get_sign()) ? 1 : 0;
        
        // Align Little Mantissa 
        uint16_t Shifted_LitMan = LitMan >> Exp_diff;
        uint16_t Round_Bit = (Exp_diff == 0) || (Exp_diff > 11) ? 0 : ((LitMan >> (Exp_diff-1)) & 0x1);
        uint16_t Sticky_Bits = (LitMan & ((0x1 << Exp_diff) - 0x1));
        uint16_t Sticky_Bit = 0;
        for(uint16_t i=0;i<16;i++) if((Sticky_Bits>>i) & 0x1 == 1) Sticky_Bit = 1;

        #ifdef PRINT_DEBUG
            std::cout<<" Compare Exponents and Align Mantissa"<<std::endl;
            std::cout<<" is_both_sub     : "<<std::hex<<is_both_sub<<std::endl;
            std::cout<<" RawExp          : "<<std::hex<<RawExp<<std::endl;
            std::cout<<" BigMan          : "<<std::hex<<BigMan<<std::endl;                                                 
            std::cout<<" LitMan          : "<<std::hex<<LitMan<<std::endl;                                                 
            std::cout<<" Exp_diff        : "<<std::hex<<Exp_diff<<std::endl;                                                 
            std::cout<<" Shifted_LitMan  : "<<std::hex<<Shifted_LitMan<<std::endl;                                                 
            std::cout<<" Round_Bit       : "<<std::hex<<Round_Bit<<std::endl;  
            std::cout<<" Sticky_Bit      : "<<std::hex<<Sticky_Bit<<std::endl;  
        #endif
        // Mantissa Addition (Both Sigh are same)
        // get_sign() == other.get_sign()

        uint16_t shifted_Bigman = BigMan<<2;
        uint16_t Shifted_Shifted_LitMan = Shifted_LitMan<<2;
        uint16_t Shifted_Round_Bit = Round_Bit<<1;
        uint16_t RawManAdd;
        if(is_Add == 1) {
            RawManAdd = shifted_Bigman + Shifted_Shifted_LitMan + Shifted_Round_Bit + Sticky_Bit;
        } else {
            uint16_t sub_LitMan = ~(Shifted_Shifted_LitMan + Shifted_Round_Bit + Sticky_Bit) + 1;
            RawManAdd = shifted_Bigman + sub_LitMan;
        }
        #ifdef PRINT_DEBUG
            std::cout<<" Mantissa Addition"<<std::endl;
            std::cout<<" RawManAdd          : "<<std::hex<<RawManAdd<<std::endl;
        #endif
        // Post-Normalization 

        uint16_t lzd = 0; // Leading-zero Detector 
        if(is_Add != 1) {
            //  RawManAdd[12:0]
            for(uint16_t i=0;i<13;i++) {
                uint16_t one_bit = (uint16_t)(RawManAdd>>i) & 0x1;
                if(one_bit == 0x1) lzd = i;
            }
        }
        uint16_t rshit =(uint16_t)(12-lzd);
        uint16_t rshifted_RawManAdd = (uint16_t)(RawManAdd << rshit);
        uint16_t is_msb_13 = ((RawManAdd >> 13) == 1) ? 1 : 0;

        uint16_t GRS_bits;
        if(is_msb_13) {            
            uint16_t RawManAdd_Sticky_Bit = 0;
            uint16_t RawManAdd_GR_Bits = ((RawManAdd >>2) & 0x3) << 1;
            for(uint16_t i=0;i<2;i++) if((Sticky_Bits>>i) & 0x1 == 1) RawManAdd_Sticky_Bit = 1;            
            GRS_bits = RawManAdd_GR_Bits + RawManAdd_Sticky_Bit;
            #ifdef PRINT_DEBUG
                std::cout<<" RawManAdd_Sticky_Bit: "<<std::hex<<RawManAdd_Sticky_Bit<<std::endl;
                std::cout<<" RawManAdd_GR_Bits   : "<<std::hex<<RawManAdd_GR_Bits<<std::endl;
            #endif
        } else {
            if(is_Add == 1 || is_both_sub) GRS_bits = RawManAdd & 0x7;
            else                           GRS_bits = rshifted_RawManAdd & 0x7;
        }

        uint16_t RoundUp = (GRS_bits == 3 || GRS_bits == 6 || GRS_bits == 7) ? 1 : 0;
        uint16_t PostNormMan = (is_msb_13 == 1)                ? (uint16_t)(RawManAdd >> 3) + RoundUp :
                               (is_Add == 1 || is_both_sub)    ? (uint16_t)(RawManAdd >> 2) + RoundUp :
                                                                 (uint16_t)(rshifted_RawManAdd >> 2) + RoundUp;                                                

        uint16_t NormMan = (((PostNormMan >> 11) == 1) ?  PostNormMan >> 1 : PostNormMan) & M_MASK;
        uint16_t NormExp = (uint16_t)(RawExp + is_msb_13 + ((PostNormMan >> 11) == 1)) + (uint16_t)(is_both_sub & ((PostNormMan >> 10) == 1));
        uint16_t CalOut = SignOut << (E_WIDTH + M_WIDTH) | NormExp << M_WIDTH | NormMan;

        #ifdef PRINT_DEBUG
            std::cout<<" Post Normalization"<<std::endl;
        
            std::cout<<" lzd                : "<<std::hex<<lzd<<std::endl;   
            std::cout<<" rshit              : "<<std::hex<<rshit<<std::endl;   
            std::cout<<" rshifted_RawManAdd : "<<std::hex<<rshifted_RawManAdd<<std::endl;   
            std::cout<<" is_msb_13          : "<<std::hex<<is_msb_13<<std::endl;
            std::cout<<" GRS_bits           : "<<std::hex<<GRS_bits<<std::endl;
            std::cout<<" RoundUp            : "<<std::hex<<RoundUp<<std::endl;
            std::cout<<" PostNormMan        : "<<std::hex<<PostNormMan<<std::endl;
            std::cout<<" NormMan            : "<<std::hex<<NormMan<<std::endl;
            std::cout<<" NormExp            : "<<std::hex<<NormExp<<std::endl;
            std::cout<<" CalOut             : "<<std::hex<<CalOut<<std::endl;
        #endif
        // if(is_Add != 1) exit(1);
        if(PostNormMan == 0) return FP16(uint16_t(pzero));
        else if(NormExp == E_MAX && SignOut == 0) return FP16(uint16_t(pInf));
        else if(NormExp == E_MAX && SignOut == 1) return FP16(uint16_t(nInf));
        else return FP16(uint16_t(CalOut));
    }    
}

FP16 FP16::operator-(const FP16& other) const {
    return FP16(uint16_t(0));
}

FP16 FP16::operator*(const FP16& other) const {
    return FP16(uint16_t(0));
}

FP16 FP16::operator/(const FP16& other) const {
    return FP16(uint16_t(0));
}

uint16_t FP16::get_sign() const {
    uint16_t fp16_s = (value.ival >> (M_WIDTH + E_WIDTH)) & 0x1;
    return fp16_s;    
}
uint16_t FP16::get_expo() const {
    // std::cout<<"ival "<<std::hex<<value.ival<<std::endl;
    // std::cout<<"ival>>M_WIDTH "<<std::hex<<(value.ival >> M_WIDTH)<<std::endl;
    // std::cout<<"(value.ival >> M_WIDTH) & E_MASK "<<std::hex<<((value.ival >> M_WIDTH) & E_MASK)<<std::endl;
    uint16_t fp16_e = (value.ival >> M_WIDTH) & E_MASK;
    return fp16_e;
}
uint16_t FP16::get_mani() const {
    uint16_t fp16_m = value.ival & M_MASK;
    return fp16_m;
}
uint16_t FP16::get_emani() const {
    uint32_t fp16_m;
    if(is_subnor()) fp16_m = value.ival & M_MASK;
    else            fp16_m = (0x1 << M_WIDTH) | (value.ival & M_MASK);
    return fp16_m;
}
uint16_t FP16::get_round_bits(uint16_t norm_grs) const {
    return 0;
}

bool FP16::is_inf() const {
    uint16_t fp16_e = get_expo();
    uint16_t fp16_m = get_mani();

    if(fp16_e == E_MASK && fp16_m == 0) return true;
    else                                return false;
}

bool FP16::is_nan() const {
    uint16_t fp16_e = get_expo();
    uint16_t fp16_m = get_mani();

    if(fp16_e == E_MASK && fp16_m != 0) return true;
    else                                return false;
}

bool FP16::is_subnor() const {
    uint16_t fp16_e = get_expo();
    uint16_t fp16_m = get_mani();

    if(fp16_e == 0 && fp16_m != 0) return true;
    else                           return false;    
}

bool FP16::is_zero() const {
    uint16_t fp16_e = get_expo();
    uint16_t fp16_m = get_mani();

    if(fp16_e == 0 && fp16_m == 0) return true;
    else                           return false;    
}