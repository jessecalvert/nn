#pragma once

#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <float.h>
    
typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;
typedef int32 bool32;

typedef int8 s8;
typedef int16 s16;
typedef int32 s32;
typedef int64 s64;
typedef bool32 b32;

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

typedef uint8 u8;
typedef uint16 u16;
typedef uint32 u32;
typedef uint64 u64;

typedef size_t umm;
    
typedef float real32;
typedef double real64;

typedef real32 r32;
typedef real64 r64;

#define Real32Maximum FLT_MAX
#define U32MAX 4294967295
#define PI_R32 3.14159265359f
    
#define internal static
#define local_persist static
#define global_variable static

#define ArrayCount(Value) (sizeof(Value) / sizeof((Value)[0]))

#define Minimum(A, B) (A) < (B) ? (A) : (B)
#define Maximum(A, B) (A) > (B) ? (A) : (B)
#define Swap(A, B, type) type TEMP_VAR_FOR_MACRO_USE_PLS_DONT_REDEFINE = (A); (A) = (B); (B) = TEMP_VAR_FOR_MACRO_USE_PLS_DONT_REDEFINE

#if NN_INTERNAL
	#define Assert(Value) if(!(Value)) {*(int *)0 = 0;}
#else
	#define Assert(Value)
#endif
#define InvalidCodePath Assert(0)
#define InvalidDefaultCase default:{InvalidCodePath;}break

#define Kilobytes(Value) (Value)*1024LL
#define Megabytes(Value) Kilobytes(Value)*1024LL
#define Gigabytes(Value) Megabytes(Value)*1024LL
#define Terabytes(Value) Gigabytes(Value)*1024LL

// TODO: Remove this.
#include <stdio.h>

#include "nn_memory.h"
#include "nn_intrinsics.h"
#include "nn_random.h"
#include "nn_math.h"

inline void
PrintVec(vec A)
{
	printf("(");
	for(u32 Index = 0;
	    Index < A.Dimension;
	    ++Index)
	{
		if(Index > 0)
		{
			printf(", ");
		}
		r32 *Value = A.Data + Index;
		printf("%3.2f", *Value);
	}
	printf(")\n");
}

inline void
PrintMatrix(matrix A)
{
	r32 *Row = A.Data;
	for(u32 RowIndex = 0;
	    RowIndex < A.RowCount;
	    ++RowIndex)
	{
		if(A.RowCount == 1)
		{
			printf("[ ");
		}
		else if(RowIndex == 0)
		{
			printf("/ ");			
		}
		else if(RowIndex == (A.RowCount - 1))
		{
			printf("\\ ");
		}
		else
		{
			printf("| ");
		}

		r32 *Source = Row;
		for(u32 ColumnIndex = 0;
		    ColumnIndex < A.ColumnCount;
		    ++ColumnIndex)
		{
			printf("%+3.2f ", *Source);
			Source += A.RowCount;
		}
		++Row;

		if(A.RowCount == 1)
		{
			printf("]\n");
		}
		else if(RowIndex == 0)
		{
			printf("\\\n");			
		}
		else if(RowIndex == (A.RowCount - 1))
		{
			printf("/\n");
		}
		else
		{
			printf("|\n");
		}
	}
}

struct command_line_options
{
	char *LoadNetwork;
	char *SaveNetwork;

	u32 HiddenLayerNeurons;
	u32 EpochCount;
	u32 BatchSize;

	r32 LearningRate;
	r32 Regularization;
};

struct feed_forward_result
{
	vec *WeightedInputs;
	vec *Activations;
};

struct feed_forward_batch_result
{
	matrix *Activations;
	matrix *WeightedInputs;	
};

struct back_propagate_batch_result
{
	matrix *WeightedInputs;
	matrix *Activations;
	matrix *Errors;
};

struct batch
{
	matrix Input;
	matrix Output;
};

enum cost_function
{
	CostFn_Quadratic,
	CostFn_CrossEntropy,
	CostFn_Count,
};

struct neural_network
{
	cost_function CostFn;

	u32 LayerCount;
	u32 *Layers;

	matrix *WeightMatrices;
	vec *BiasVectors;
};

struct data_set
{
	u32 DataCount;
	vec *InputData;
	vec *OutputData;
};

#include "nn_io.h"

internal void
PrintNeuralNetwork(neural_network Network)
{
	printf("[");
	for(u32 Index = 0;
	    Index < Network.LayerCount;
	    ++Index)
	{
		if(Index > 0)
		{
			printf(", ");
		}

		u32 Value = Network.Layers[Index];
		printf("%d", Value);
	}
	printf("]\n");

	for(u32 Index = 1;
	    Index < Network.LayerCount;
	    ++Index)
	{
		printf("WeightMatrix %d->%d:\n", Index - 1, Index);
		PrintMatrix(Network.WeightMatrices[Index]);
		printf("BiasVector %d:\n", Index);
		PrintVec(Network.BiasVectors[Index]);
	}
}

inline b32
StringCompare(char *A, char *B)
{
	b32 Result = true;
	while(*A && *B)
	{
		if(*A != *B)
		{
			Result = false;
			break;
		}
		++A;
		++B;
	}

	Result = (*A == *B);
	return Result;
}