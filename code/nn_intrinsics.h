#pragma once

// TODO: Remove this.
#include <math.h>

inline r32
U8ToR32(u8 Value)
{
	r32 Result = (r32)Value / 255.0f;
	return Result;
}

inline r32
Exp(r32 Value)
{
	r32 Result = (r32)exp(Value);
	return Result;
}

inline r32
Sigmoid(r32 Value)
{
	r32 Result = 1.0f / (1.0f + Exp(-Value));
	return Result;
}

inline r32
Square(r32 Value)
{
	r32 Result = Value*Value;
	return Result;
}

inline r32
SigmoidPrime(r32 Value)
{
	r32 SigmoidValue = Sigmoid(Value);
	r32 Result = SigmoidValue * (1.0f - SigmoidValue);
	return Result;
}

inline r32
Ln(r32 Value)
{
	r32 Result = logf(Value);
	return Result;
}

inline r32
SquareRoot(r32 Value)
{
	r32 Result = sqrtf(Value);
	return Result;
}