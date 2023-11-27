#ifndef FP16G_H
#define FP16G_H

#include "half.hpp"

typedef half_float::half fp16;

union fp16i
{
    fp16 fval;
    uint16_t ival;

    fp16i()
    {
        ival = 0;
    }
    fp16i(fp16 x)
    {
        fval = x;
    }
    fp16i(uint16_t x)
    {
        ival = x;
    }
};    

struct FP16g {
    fp16 value;

    FP16g(fp16 val);

    FP16g operator+(const FP16g& other) const;
    FP16g operator-(const FP16g& other) const;
    FP16g operator*(const FP16g& other) const;
    FP16g operator/(const FP16g& other) const;

    float toFloat() const;

    static FP16g fromFloat(float fp32);
};

#endif  // FP16_H