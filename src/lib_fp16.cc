#ifndef LIB_FP16_H
#define LIB_FP16_H

// #define PRINT_DEBUG 1
#include <stdint.h> 
#include <cstdio>
#include "svdpi.h"

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

union fp32f
{
    float fval;
    uint32_t ival;
};  
union fp32i
{
    uint32_t ival;
    float fval;
};  

uint16_t numpy_floatbits_to_halfbits(uint32_t f) {
  uint16_t h_sgn = (uint16_t)((f & 0x80000000u) >> 16);
  uint32_t f_exp = f & 0x7f800000u;
  uint32_t f_sig = f & 0x007fffffu;

  // Exponent overflow/NaN converts to signed inf/NaN
  if (f_exp >= 0x47800000u) {
    if ((f_exp == 0x7f800000u) && (f_sig != 0)) {
      // NaN - propagate the flag in the significand...
      uint16_t ret = (uint16_t)(0x7c00u + (f_sig >> 13));
      ret += (ret == 0x7c00u); // ...but make sure it stays a NaN
      return h_sgn + ret;
    } else {
      // (overflow to) signed inf
      return (uint16_t)(h_sgn + 0x7c00u);
    }
  }

  // Exponent underflow converts to a subnormal half or signed zero
  if (f_exp <= 0x38000000u) {
    // Signed zeros, subnormal floats, and floats with small
    // exponents all convert to signed zero half-floats.
    if (f_exp < 0x33000000u) {
      return h_sgn;
    }
    // Make the subnormal significand
    f_exp >>= 23;
    f_sig += 0x00800000u;
    f_sig >>= (113 - f_exp);
    // Handle rounding by adding 1 to the bit beyond half precision
    //
    // If the last bit in the half significand is 0 (already even),
    // and the remaining bit pattern is 1000...0, then we do not add
    // one to the bit after the half significand. However, the
    // (113 - f_exp) shift can lose up to 11 bits, so the || checks
    // them in the original. In all other cases, we can just add one.
    if (((f_sig & 0x3fffu) != 0x1000u) || (f & 0x07ffu)) {
      f_sig += 0x1000u;
    }
    uint16_t h_sig = (uint16_t)(f_sig >> 13);
    // If the rounding causes a bit to spill into h_exp, it will
    // increment h_exp from zero to one and h_sig will be zero.
    // This is the correct result.
    return (uint16_t)(h_sgn + h_sig);
  }

  // Regular case with no overflow or underflow
  uint16_t h_exp = (uint16_t)((f_exp - 0x38000000u) >> 13);
  // Handle rounding by adding 1 to the bit beyond half precision
  //
  // If the last bit in the half significand is 0 (already even), and
  // the remaining bit pattern is 1000...0, then we do not add one to
  // the bit after the half significand. In all other cases, we do.
  if ((f_sig & 0x3fffu) != 0x1000u) {
      f_sig += 0x1000u;
  }
  uint16_t h_sig = (uint16_t)(f_sig >> 13);
  // If the rounding causes a bit to spill into h_exp, it will
  // increment h_exp by one and h_sig will be zero. This is the
  // correct result. h_exp may increment to 15, at greatest, in
  // which case the result overflows to a signed inf.

  return (uint16_t)(h_sgn + h_exp + h_sig);
}

uint32_t npy_halfbits_to_floatbits(uint16_t h)
{
    uint16_t h_exp, h_sig;
    uint16_t f_sgn, f_exp, f_sig;

    h_exp = ((uint32_t)h&0x7c00u);
    f_sgn = ((uint32_t)h&0x8000u) << 16;
    switch (h_exp) {
        case 0x0000u: /* 0 or subnormal */
            h_sig = (h&0x03ffu);
            /* Signed zero */
            if (h_sig == 0) {
                return f_sgn;
            }
            /* Subnormal */
            h_sig <<= 1;
            while ((h_sig&0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            f_exp = ((uint32_t)(127 - 15 - h_exp)) << 23;
            f_sig = ((uint32_t)(h_sig&0x03ffu)) << 13;
            return (uint32_t)(f_sgn + f_exp + f_sig);
        case 0x7c00u: /* inf or NaN */
            /* All-ones exponent and a copy of the significand */
            return (uint32_t)(f_sgn + 0x7f800000u + (((uint32_t)(h&0x03ffu)) << 13));
        default: /* normalized */
            /* Just need to adjust the exponent and shift */
            return (uint32_t)(f_sgn + (((uint32_t)(h&0x7fffu) + 0x1c000u) << 13));
    }
};

uint16_t fp16_add(uint16_t fp16_a, uint16_t fp16_b) {
    uint16_t fp16_a_s = (fp16_a >> (M_WIDTH + E_WIDTH)) & 0x1;
    uint16_t fp16_b_s = (fp16_b >> (M_WIDTH + E_WIDTH)) & 0x1;
    uint16_t fp16_a_e = (fp16_a >> M_WIDTH) & E_MASK;
    uint16_t fp16_b_e = (fp16_b >> M_WIDTH) & E_MASK;
    uint16_t fp16_a_m = fp16_a & M_MASK;
    uint16_t fp16_b_m = fp16_b & M_MASK;

    uint16_t fp16_a_inf = (fp16_a_e == E_MASK && fp16_a_m == 0) ? 1 : 0;
    uint16_t fp16_b_inf = (fp16_b_e == E_MASK && fp16_b_m == 0) ? 1 : 0;
    uint16_t fp16_a_nan = (fp16_a_e == E_MASK && fp16_a_m != 0) ? 1 : 0;
    uint16_t fp16_b_nan = (fp16_b_e == E_MASK && fp16_b_m != 0) ? 1 : 0;
    uint16_t fp16_a_subnor = (fp16_a_e == 0 && fp16_a_m != 0) ? 1 : 0;
    uint16_t fp16_b_subnor = (fp16_b_e == 0 && fp16_b_m != 0) ? 1 : 0;          
    uint16_t fp16_a_zero = (fp16_a_e == 0 && fp16_a_m == 0) ? 1 : 0;
    uint16_t fp16_b_zero = (fp16_b_e == 0 && fp16_b_m == 0) ? 1 : 0;              

    uint16_t fp16_a_em = (fp16_a_subnor == 1) ? fp16_a_m : ((0x1 << M_WIDTH) | fp16_a_m);
    uint16_t fp16_b_em = (fp16_b_subnor == 1) ? fp16_b_m : ((0x1 << M_WIDTH) | fp16_b_m);

    #ifdef PRINT_DEBUG
    printf(" FP16 Addition (A+B)\n");
    printf(" Input A: 0x%04X\n",fp16_a);
    printf("  - Sign               : %x\n",fp16_a_s);
    printf("  - Exponent           : 0x%02x\n",fp16_a_e);
    printf("  - Mantissa           : 0x%03x (0x%03x)\n",fp16_a_m,fp16_a_em);
    printf("  - Inf/NaN/SubNor/Zero: %0x/%0x/%0x/%0x\n",fp16_a_inf,fp16_a_nan,fp16_a_subnor,fp16_a_zero);

    printf(" Input B: 0x%04X\n",fp16_b);
    printf("  - Sign               : %x\n",fp16_b_s);
    printf("  - Exponent           : 0x%02x\n",fp16_b_e);
    printf("  - Mantissa           : 0x%03x (0x%03x)\n",fp16_b_m,fp16_b_em);
    printf("  - Inf/NaN/SubNor/Zero: %0x/%0x/%0x/%0x\n",fp16_b_inf,fp16_b_nan,fp16_b_subnor,fp16_b_zero);
    #endif 
    // Compare Exponents & Mantissa 
    uint16_t is_both_subnormal = (fp16_a_subnor == 1 && fp16_b_subnor == 1) ? 1 : 0;
    uint16_t is_large_a_exp = (fp16_a_e > fp16_b_e)   ? 1 : 0;       // Compare Exponent for Input A and Input B
    uint16_t is_same_exp    = (fp16_a_e == fp16_b_e)  ? 1 : 0;       // Compare Exponent for Input A and Input B
    uint16_t is_large_a_man = (fp16_a_em > fp16_b_em) ? 1 : 0;
    uint16_t is_large_a     = (is_large_a_exp == 1)   ? 1 : 
                              ((is_same_exp == 1)     ? ((is_large_a_man == 1) ? 1 : 0) : 0);

    uint16_t RawExp     = (is_large_a_exp == 1) ? fp16_a_e : fp16_b_e;
    uint16_t BigMan     = ((is_large_a_exp == 1) || (is_same_exp && is_large_a_man)) ? fp16_a_em : fp16_b_em;
    uint16_t LitMan     = ((is_large_a_exp == 1) || (is_same_exp && is_large_a_man)) ? fp16_b_em : fp16_a_em;

    uint16_t compensated_Exp = (fp16_a_subnor != fp16_b_subnor) ? 1 : 0;
    uint16_t Exp_diff  = (is_large_a_exp == 1) ? (fp16_a_e - fp16_b_e) :
                                                 (fp16_b_e - fp16_a_e);
    Exp_diff = Exp_diff - compensated_Exp;
    uint16_t SignOut = (is_large_a == 1)      ? fp16_a_s : fp16_b_s;
    uint16_t is_Add  = (fp16_a_s == fp16_b_s) ? 1        : 0;
    
    // Align Little Mantissa 
    uint16_t Shifted_LitMan = LitMan >> Exp_diff;
    uint16_t Round_Bit = (Exp_diff == 0) || (Exp_diff > 11) ? 0 : ((LitMan >> (Exp_diff-1)) & 0x1);
    uint16_t Sticky_Bits = (LitMan & ((0x1 << Exp_diff) - 0x1));
    uint16_t Sticky_Bit = 0;
    for(uint16_t i=0;i<16;i++) if((Sticky_Bits>>i) & 0x1 == 1) Sticky_Bit = 1;

    #ifdef PRINT_DEBUG
    printf(" Compare Exponents and Align Mantissa\n");
    printf("  - is_both_sub     : %x\n",is_both_subnormal);
    printf("  - RawExp          : %x\n",RawExp);
    printf("  - BigMan          : %x\n",BigMan);                                                 
    printf("  - LitMan          : %x\n",LitMan);                                                 
    printf("  - Exp_diff        : %x\n",Exp_diff);                                                 
    printf("  - Shifted_LitMan  : %x\n",Shifted_LitMan);                                                 
    printf("  - Round_Bit       : %x\n",Round_Bit);  
    printf("  - Sticky_Bit      : %x\n",Sticky_Bit);
    #endif

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
    printf(" Mantissa Addition\n");
    printf("  - RawManAdd : %x\n",RawManAdd);
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
            printf("  - RawManAdd_Sticky_Bit : %x\n",RawManAdd_Sticky_Bit);
            printf("  - RawManAdd_GR_Bits    : %x\n",RawManAdd_GR_Bits);
        #endif
    } else {
        if(is_Add == 1 || is_both_subnormal) GRS_bits = RawManAdd & 0x7;
        else                                 GRS_bits = rshifted_RawManAdd & 0x7;
    }

    uint16_t RoundUp = (GRS_bits == 3 || GRS_bits == 6 || GRS_bits == 7) ? 1 : 0;
    uint16_t PostNormMan = (is_msb_13 == 1)                       ? (uint16_t)(RawManAdd >> 3) + RoundUp :
                            (is_Add == 1 || is_both_subnormal)    ? (uint16_t)(RawManAdd >> 2) + RoundUp :
                                                                    (uint16_t)(rshifted_RawManAdd >> 2) + RoundUp;                                                

    uint16_t NormMan = (((PostNormMan >> 11) == 1) ?  PostNormMan >> 1 : PostNormMan) & M_MASK;
    uint16_t NormExp = (uint16_t)(RawExp + is_msb_13 + ((PostNormMan >> 11) == 1)) + (uint16_t)(is_both_subnormal & ((PostNormMan >> 10) == 1));
    uint16_t CalOut = SignOut << (E_WIDTH + M_WIDTH) | NormExp << M_WIDTH | NormMan;


    #ifdef PRINT_DEBUG
    printf("  - lzd                : %x\n",lzd);
    printf("  - rshit              : %x\n",rshit);
    printf("  - rshifted_RawManAdd : %x\n",rshifted_RawManAdd);
    printf("  - is_msb_13          : %x\n",is_msb_13);
    printf("  - GRS_bits           : %x\n",GRS_bits);
    printf("  - RoundUp            : %x\n",RoundUp);
    printf("  - PostNormMan        : %x\n",PostNormMan);
    printf("  - NormMan            : %x\n",NormMan);
    printf("  - NormExp            : %x\n",NormExp);
    printf("  - CalOut             : %x\n",CalOut);
    #endif
    if(fp16_a_nan == 1 || fp16_b_nan == 1) 
        return uint16_t(qNaN);
    else if((fp16_a_inf == 1 && fp16_b_inf == 1) && (fp16_a_s != fp16_b_s)) 
        return uint16_t(qNaN);
    else if((fp16_a_inf == 1 && fp16_a_s == 0) || (fp16_b_inf == 1 && fp16_a_s == 0)) 
        return uint16_t(pInf);
    else if((fp16_a_inf == 1 && fp16_a_s == 1) || (fp16_b_inf == 1 && fp16_a_s == 1)) 
        return uint16_t(nInf);
    else if(PostNormMan == 0) 
        return uint16_t(pzero);
    else if(NormExp == E_MAX && SignOut == 0) 
        return uint16_t(pInf);
    else if(NormExp == E_MAX && SignOut == 1) 
        return uint16_t(nInf);
    else if(fp16_a_zero == 1 || fp16_b_zero == 1) {
        if(fp16_a_zero == 1) return fp16_b;
        else                 return fp16_a;
    } 
    else return uint16_t(CalOut);
    
    // return fp16_c;
}


DPI_DLLESPEC
int fp16_add_int(int int_a, int int_b) {
    uint16_t fp16_a = int_a & 0xFFFF;
    uint16_t fp16_b = int_a & 0xFFFF;
    uint16_t fp16_c = fp16_add(fp16_a,fp16_b);
    return (int)fp16_c;
}

DPI_DLLESPEC
int FP32_to_FP16(float fp32_value){
    union fp32f tmp = {fp32_value};
    uint16_t uint16_value = numpy_floatbits_to_halfbits(tmp.ival);
    return (int)uint16_value;
};

DPI_DLLESPEC
float FP16_to_FP32(int fp16_value){
    uint16_t tmp = fp16_value & 0xffff;
    uint32_t fp32_int = npy_halfbits_to_floatbits(tmp);
    union fp32i union_tmp = {fp32_int};
    // uint16_t uint16_value = numpy_floatbits_to_halfbits(tmp.ival);
    return union_tmp.fval;
};

#endif