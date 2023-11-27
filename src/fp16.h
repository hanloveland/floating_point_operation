#ifndef FP16_H
#define FP16_H

#include "half.hpp"

typedef half_float::half fp16;

union fp16_u
{
    fp16 fval;
    uint16_t ival;

    fp16_u()
    {
        ival = 0;
    }
    fp16_u(float x)
    {
        fval = fp16(x);
    }    
    fp16_u(fp16 x)
    {
        fval = x;
    }
    fp16_u(uint16_t x)
    {
        ival = x;
    }
};  

struct FP16 {
    union fp16_u value;

    FP16(fp16_u uvalue);
    FP16(float fvalue);
    FP16(uint16_t ivalue);

    uint16_t get_sign() const;
    uint16_t get_expo() const;
    uint16_t get_mani() const;
    uint16_t get_emani() const; //get mantissa acoording whether it's subnormal or not. 
    uint16_t get_round_bits(uint16_t norm_grs) const;

    bool is_inf() const;
    bool is_nan() const;
    bool is_subnor() const;
    bool is_zero() const;
    
    FP16 operator+(const FP16& other) const;
    FP16 operator-(const FP16& other) const;
    FP16 operator*(const FP16& other) const;
    FP16 operator/(const FP16& other) const;

};

#endif