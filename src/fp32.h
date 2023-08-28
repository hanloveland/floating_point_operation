#ifndef FP32_H
#define FP32_H

#include <cstdint>

union fp32_u
{
    float fval;
    uint32_t ival;

    fp32_u()
    {
        ival = 0;
    }
    fp32_u(float x)
    {
        fval = x;
    }
    fp32_u(uint32_t x)
    {
        ival = x;
    }
};  

struct FP32 {
    union fp32_u value;

    FP32(fp32_u uvalue);
    FP32(float fvalue);
    FP32(uint32_t ivalue);

    uint32_t get_sign() const;
    uint32_t get_expo() const;
    uint32_t get_mani() const;
    uint32_t get_emani() const; //get mantissa acoording whether it's subnormal or not. 
    uint32_t get_round_bits(uint32_t norm_grs) const;

    bool is_inf() const;
    bool is_nan() const;
    bool is_subnor() const;
    bool is_zero() const;
    
    FP32 operator+(const FP32& other) const;
    FP32 operator-(const FP32& other) const;
    FP32 operator*(const FP32& other) const;
    FP32 operator/(const FP32& other) const;

};

#endif