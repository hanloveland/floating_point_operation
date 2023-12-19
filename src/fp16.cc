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
#define sNaN 0x7FFF //sNaN
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
            std::cout<<"return sNaN"<<std::endl;
        #endif
        return FP16(uint16_t(sNaN));     
    } else if(is_inf() || other.is_inf()) { // Infinity Arthmetic
        if((is_inf() && get_sign() == 0) || (other.is_inf() && other.get_sign() == 0)) return FP16(uint16_t(pInf));  
        else                                                                           return FP16(uint16_t(nInf));  
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
        uint16_t is_large_a_exp = get_expo() > other.get_expo()   ? 1 : 0;        // Compare Exponent for Input A and Input B
        uint16_t is_same_exp    = get_expo() == other.get_expo()  ? 1 : 0;        // Compare Exponent for Input A and Input B
        uint16_t is_large_a_man = get_emani() > other.get_emani() ? 1 : 0;        // Compare Mantissa for Input A and Input B
        uint16_t is_large_a     = is_large_a_exp == 1 ? 1 : (is_same_exp == 1 ? (is_large_a_man == 1 ? 1 : 0) : 0);

        uint16_t RawExp     = (is_large_a_exp == 1) ? get_expo() : other.get_expo();
        uint16_t BigMan     = ((is_large_a_exp == 1) || (is_same_exp && is_large_a_man)) ? get_emani() : other.get_emani();
        uint16_t LitMan     = ((is_large_a_exp == 1) || (is_same_exp && is_large_a_man)) ? other.get_emani() : get_emani();

        // if one of operand is subnormal, decrease the exponent difference by one
        // e.g., exponent is zero (subnormal), this mean exponent is same with exponent one 
        uint16_t compensated_Exp = (is_subnor() ^ other.is_subnor()) ? 1 : 0;
        uint16_t Exp_diff  = (is_large_a_exp == 1) ? (get_expo() - other.get_expo()) :
                                                     (other.get_expo() - get_expo());
        Exp_diff = Exp_diff - compensated_Exp;
        RawExp = Exp_diff > 3 ? (RawExp-3) : (RawExp - Exp_diff);
        
        // Maximum Left Shift for BigMant is 3
        // Exp_Diff 1 ~ 3: Shift Left, Exp_Diff > 3: Shift Right 
        uint16_t LShift  = (Exp_diff > 3) ? 3          : Exp_diff;
        uint16_t RShift  = (Exp_diff > 3) ? Exp_diff-3 : 0;
        // SignOut is same with large operand
        uint16_t SignOut = is_large_a == 1 ? get_sign() : other.get_sign();
        // if both operands are same, do addition, else subtraction
        uint16_t is_Add = (get_sign() == other.get_sign()) ? 1 : 0;

        // Align Big/Little Mantissa 
        uint16_t Shifted_BigMan = (uint16_t)(BigMan << LShift);
        uint16_t Shifted_LitMan = (uint16_t)(LitMan >> (RShift+2));
        uint16_t Round_Bit = (Exp_diff <= 3) ? ((LitMan >> (1)) & 0x1)   :
                             (Exp_diff > 14) ? 0                         : ((LitMan >> (RShift+1)) & 0x1);                
        uint16_t Sticky_Bits = (RShift == 0) ? (LitMan & 0x1) : (LitMan & ((0x1 << (RShift+1)) - 0x1));
        uint16_t Sticky_Bit = 0;
        for(uint16_t i=0;i<16;i++) if((Sticky_Bits>>i) & 0x1 == 1) Sticky_Bit = 1;
        Shifted_LitMan = (uint16_t)(Shifted_LitMan << 2) + (uint16_t)(Round_Bit << 1) + (uint16_t)(Sticky_Bit);
        #ifdef PRINT_DEBUG
            std::cout<<" Compare Exponents and Align Mantissa"<<std::endl;
            std::cout<<" is_both_sub     : "<<std::hex<<is_both_sub<<std::endl;
            std::cout<<" RawExp          : "<<std::hex<<RawExp<<std::endl;
            std::cout<<" BigMan          : "<<std::hex<<BigMan<<std::endl;                                                 
            std::cout<<" LitMan          : "<<std::hex<<LitMan<<std::endl;                                                 
            std::cout<<" Exp_diff        : "<<std::hex<<Exp_diff<<std::endl;                                                 
            std::cout<<" LShift          : "<<std::hex<<LShift<<std::endl;      
            std::cout<<" RShift          : "<<std::hex<<RShift<<std::endl;      
            std::cout<<" Shifted_LitMan  : "<<std::hex<<Shifted_LitMan<<std::endl;                                                 
            std::cout<<" Round_Bit       : "<<std::hex<<Round_Bit<<std::endl;  
            std::cout<<" Sticky_Bit      : "<<std::hex<<Sticky_Bit<<std::endl; 
            std::cout<<" LitMan_RoundUp  : "<<std::hex<<LitMan_RoundUp<<std::endl; 
            std::cout<<" Shifted_BigMan  : "<<std::hex<<Shifted_BigMan<<std::endl;  
            std::cout<<" Shifted_LitMan  : "<<std::hex<<Shifted_LitMan<<std::endl;              
        #endif

        // Remove last two bits to replace with Round and Sticky bit
        // uint16_t Shifted_Shifted_LitMan = Shifted_LitMan & 0x7FC;
        // uint16_t Shifted_Round_Bit = Round_Bit<<1;
        uint16_t RawManAdd;
        // Performan 16-bits addition or subtraction
        if(is_Add == 1) {
            RawManAdd = Shifted_BigMan + Shifted_LitMan;//+ Shifted_Round_Bit + Sticky_Bit;
        } else {
            uint16_t sub_LitMan = ~(Shifted_LitMan) + 1;
            RawManAdd = Shifted_BigMan + sub_LitMan;
        }
        #ifdef PRINT_DEBUG
            std::cout<<" Mantissa Addition"<<std::endl;
            std::cout<<" RawManAdd          : "<<std::hex<<RawManAdd<<std::endl;
        #endif

        // Post-Normalization 
        uint16_t lzd = 0; // Leading-zero Detector 
        for(uint16_t i=0;i<16;i++) {
            uint16_t one_bit = (uint16_t)(RawManAdd>>i) & 0x1;
            if(one_bit == 0x1) lzd = i;
        }

        uint16_t rshift = (lzd > 10) ? (uint16_t)(lzd - 10)   : (uint16_t)0;
        uint16_t lshift = (lzd < 10) ? (uint16_t)(10 - lzd)   : (uint16_t)0;
        uint16_t not_shift = (lzd == 10) ? 1 : 0;
        
        uint16_t rshifted_RawMan;
        if(rshift > 0) {
            // maximun rshift is 4 when lzd is 15
            uint16_t rshifted_rbit = (uint16_t)((RawManAdd >> (rshift-1)) & 0x1);
            uint16_t rshifted_sbits = (rshift >= 2) ? (uint16_t)(RawManAdd & (uint16_t)((0x1<<(rshift-0x1))-0x1)) : 0;
            uint16_t rshifted_sbit = 0;
            if(rshifted_sbits != 0) rshifted_sbit = 1;
            uint16_t rshifted_gbit = (uint16_t)(RawManAdd >> (rshift)) & 0x1;
            uint16_t rshifted_grs = (uint16_t)(rshifted_gbit << 2) + (uint16_t)(rshifted_rbit << 1) + (uint16_t)rshifted_sbit;
            uint16_t RoundUp = (rshifted_grs == 3 || rshifted_grs == 6 || rshifted_grs == 7) ? 1 : 0;
            rshifted_RawMan  = (uint16_t)(RawManAdd >> rshift);
            rshifted_RawMan = rshifted_RawMan + RoundUp;
            #ifdef PRINT_DEBUG
                std::cout<<" (RShift)rshifted_rbit      : "<<std::hex<<rshifted_rbit<<std::endl;
                std::cout<<" (RShift)rshifted_sbits_mask : "<<std::hex<<(uint16_t)((0x1<<(rshift-0x1))-0x1)<<std::endl;
                std::cout<<" (RShift)rshifted_sbits_mask : "<<std::hex<<(uint16_t)(RawManAdd & (uint16_t)((0x1<<(rshift-0x1))-0x1))<<std::endl;
                std::cout<<" (RShift)rshifted_sbits     : "<<std::hex<<rshifted_sbits<<std::endl;
                std::cout<<" (RShift)rshifted_sbit      : "<<std::hex<<rshifted_sbit<<std::endl;
                std::cout<<" (RShift)rshifted_grs       : "<<std::hex<<rshifted_grs<<std::endl;
                std::cout<<" (RShift)RoundUp            : "<<std::hex<<RoundUp<<std::endl;
                std::cout<<" (RShift)rshifted_RawMan    : "<<std::hex<<rshifted_RawMan<<std::endl;
            #endif
        }
        uint16_t rshifted_exp_up = 0;
        if((rshifted_RawMan >> 11) == 0x1) {
            rshifted_exp_up = 1;
            rshifted_RawMan = rshifted_RawMan >> 1;
        }
        uint16_t lshifted_RawMan = (uint16_t)(RawManAdd << lshift);
        uint16_t shifted_RawManAdd = (not_shift == 1) ? RawManAdd : (rshift > 0)      ? rshifted_RawMan       
                                                                  : (RawExp > lshift) ? lshifted_RawMan
                                                                  : (RawExp > 1)      ? (uint16_t)(RawManAdd << (RawExp-1))
                                                                                      : RawManAdd;                                                                            
        // uint16_t is_msb_13 = ((RawManAdd >> 13) == 1) ? 1 : 0;

        // uint16_t GRS_bits;
        // if(is_msb_13) {            
        //     uint16_t RawManAdd_Sticky_Bit = 0;
        //     uint16_t RawManAdd_GR_Bits = ((RawManAdd >>2) & 0x3) << 1;
        //     for(uint16_t i=0;i<2;i++) if((Sticky_Bits>>i) & 0x1 == 1) RawManAdd_Sticky_Bit = 1;            
        //     GRS_bits = RawManAdd_GR_Bits + RawManAdd_Sticky_Bit;
        //     #ifdef PRINT_DEBUG
        //         std::cout<<" RawManAdd_Sticky_Bit: "<<std::hex<<RawManAdd_Sticky_Bit<<std::endl;
        //         std::cout<<" RawManAdd_GR_Bits   : "<<std::hex<<RawManAdd_GR_Bits<<std::endl;
        //     #endif
        // } else {
        //     if(is_Add == 1 || is_both_sub) GRS_bits = RawManAdd & 0x7;
        //     else                           GRS_bits = lshifted_RawManAdd & 0x7;
        // }

        // uint16_t RoundUp = (GRS_bits == 3 || GRS_bits == 6 || GRS_bits == 7) ? 1 : 0;
        // uint16_t PostNormMan = (is_msb_13 == 1)                ? (uint16_t)(RawManAdd >> 3) + RoundUp :
        //                        (is_Add == 1 || is_both_sub)    ? (uint16_t)(RawManAdd >> 2) + RoundUp :
        //                                                          (uint16_t)(lshifted_RawManAdd >> 2) + RoundUp;    

        // Subnormal Case
        // RawExp > lshift --> RawExp --> 0
        // RawExp == 0 && lzd == A -> RawEXp --> 1
        uint16_t NormExp = (not_shift == 1) ? (RawExp == 0 ? 1 : RawExp) : (rshift > 0) ? (RawExp + rshift + rshifted_exp_up) : (RawExp > lshift) ? (RawExp - lshift) : 0;
        uint16_t NormMan = shifted_RawManAdd & M_MASK;

        // uint16_t NormMan = (((PostNormMan >> 11) == 1) ?  PostNormMan >> 1 : PostNormMan) & M_MASK;
        // uint16_t NormExp = (uint16_t)(RawExp + is_msb_13 + ((PostNormMan >> 11) == 1)) + (uint16_t)(is_both_sub & ((PostNormMan >> 10) == 1)) - (is_Add ? 0 : lshit);
        uint16_t CalOut = SignOut << (E_WIDTH + M_WIDTH) | NormExp << M_WIDTH | NormMan;

        #ifdef PRINT_DEBUG
            std::cout<<" Post Normalization"<<std::endl;
        
            std::cout<<" lzd                : "<<std::hex<<lzd<<std::endl;               
            std::cout<<" not_shift          : "<<std::hex<<not_shift<<std::endl;   
            std::cout<<" rshift             : "<<std::hex<<rshift<<std::endl;   
            std::cout<<" lshift             : "<<std::hex<<lshift<<std::endl;   
            std::cout<<" rshifted_RawMan    : "<<std::hex<<rshifted_RawMan<<std::endl;   
            std::cout<<" lshifted_RawMan    : "<<std::hex<<lshifted_RawMan<<std::endl;   
            std::cout<<" shifted_RawManAdd  : "<<std::hex<<shifted_RawManAdd<<std::endl;   
            std::cout<<" NormMan            : "<<std::hex<<NormMan<<std::endl;
            std::cout<<" NormExp            : "<<std::hex<<NormExp<<std::endl;
            std::cout<<" CalOut             : "<<std::hex<<CalOut<<std::endl;
        #endif
        if(RawManAdd == 0)                        return FP16(uint16_t(pzero));
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