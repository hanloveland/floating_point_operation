#include "FP16G.h"

FP16g::FP16g(fp16 val) : value(val) {}

FP16g FP16g::operator+(const FP16g& other) const {
    fp16 sum = value + other.value;
    return FP16g(sum);
}

FP16g FP16g::operator-(const FP16g& other) const {
    fp16 diff = value - other.value;
    return FP16g(diff);
}

FP16g FP16g::operator*(const FP16g& other) const {
    fp16 product = value * other.value;
    return FP16g(product);
}

FP16g FP16g::operator/(const FP16g& other) const {
    fp16 div = value / other.value;
    return FP16g(div);
}

float FP16g::toFloat() const {
    float fp32 = value;
    return fp32; 
}

FP16g FP16g::fromFloat(float fp32) {
    fp16 val0(fp32);
    return val0;
}