#include "fp16.h"
#include <iostream>

// #define PRINT_DEBUG 1
#define K_WIDTH 16
#define E_WIDTH 5
#define M_WIDTH 11
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
#define DIFF_THR 23
// Use RoundTiesToEven Rounding


FP16::FP16(fp16_u uvalue) : value(uvalue) {}
FP16::FP16(float fvalue) : value(fp16_u(fvalue)) {}
FP16::FP16(uint16_t ivalue) : value(fp16(ivalue)) {}

FP16 FP16::operator+(const FP16& other) const {
    return FP16(uint16_t(0));
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