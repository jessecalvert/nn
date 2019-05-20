
#pragma once

struct vec
{
	r32 *Data;
	u32 Dimension;
};

inline vec
VecRaw_(memory_pool *Pool, u32 Dimension)
{
	vec Result = {};
	Result.Data = PoolPushArray(Pool, r32, Dimension);
	Result.Dimension = Dimension;
	return Result;
}

inline vec
Vec(memory_pool *Pool, r32 *Data, u32 Dimension)
{
	vec Result = VecRaw_(Pool, Dimension);

	for(u32 Index = 0;
	    Index < Result.Dimension;
	    ++Index)
	{
		r32 *Value = Result.Data + Index;
		*Value = *(Data + Index);
	}

	return Result;
}

inline vec
Vec(r32 *Data, u32 Dimension)
{
	vec Result = {};
	Result.Data = Data;
	Result.Dimension = Dimension;
	return Result;
}

inline vec
VecRand(memory_pool *Pool, u32 Dimension, r32 Mean, r32 StandardDeviation)
{
	vec Result = VecRaw_(Pool, Dimension);

	for(u32 Index = 0;
	    Index < Result.Dimension;
	    ++Index)
	{
		r32 *Value = Result.Data + Index;
		*Value = RandomGaussian(Mean, StandardDeviation);
	}

	return Result;	
}

inline vec
VecZero(memory_pool *Pool, u32 Dimension)
{
	vec Result = VecRaw_(Pool, Dimension);

	for(u32 Index = 0;
	    Index < Result.Dimension;
	    ++Index)
	{
		r32 *Value = Result.Data + Index;
		*Value = 0.0f;
	}

	return Result;
}

inline vec
Hadamard(memory_pool *Pool, vec A, vec B)
{
	Assert(A.Dimension == B.Dimension);

	vec Result = VecRaw_(Pool, A.Dimension);

	for(u32 Index = 0;
	    Index < Result.Dimension;
	    ++Index)
	{
		r32 *Value = Result.Data + Index;
		*Value = *(A.Data + Index) * *(B.Data + Index);
	}

	return Result;
}

inline vec
Plus(memory_pool *Pool, vec A, vec B)
{
	Assert(A.Dimension == B.Dimension);
	
	vec Result = VecRaw_(Pool, A.Dimension);
	
	for(u32 Index = 0;
	    Index < Result.Dimension;
	    ++Index)
	{
		r32 *Value = Result.Data + Index;
		*Value = *(A.Data + Index) + *(B.Data + Index);
	}

	return Result;
}

inline vec
Minus(memory_pool *Pool, vec A, vec B)
{
	Assert(A.Dimension == B.Dimension);
	
	vec Result = VecRaw_(Pool, A.Dimension);
	
	for(u32 Index = 0;
	    Index < Result.Dimension;
	    ++Index)
	{
		r32 *Value = Result.Data + Index;
		*Value = *(A.Data + Index) - *(B.Data + Index);
	}

	return Result;
}

inline r32
InnerProduct(vec A, vec B)
{
	Assert(A.Dimension == B.Dimension);

	r32 Result = 0.0f;
	for(u32 Index = 0;
	    Index < A.Dimension;
	    ++Index)
	{
		r32 *AValue = A.Data + Index;
		r32 *BValue = B.Data + Index;
		Result += *AValue * *BValue;
	}

	return Result;
}

inline vec
Sigmoid(memory_pool *Pool, vec V)
{
	vec Result = VecRaw_(Pool, V.Dimension);

	r32 *NewValue = Result.Data;
	r32 *OldValue = V.Data;
	for(u32 Index = 0;
	    Index < Result.Dimension;
	    ++Index)
	{
		*NewValue++ = Sigmoid(*OldValue++);
	}

	return Result;
}

inline vec
SigmoidPrime(memory_pool *Pool, vec V)
{
	vec Result = VecRaw_(Pool, V.Dimension);

	r32 *NewValue = Result.Data;
	r32 *OldValue = V.Data;
	for(u32 Index = 0;
	    Index < Result.Dimension;
	    ++Index)
	{
		*NewValue++ = SigmoidPrime(*OldValue++);
	}

	return Result;
}

inline void
VectorScaleEquals(r32 Scale, vec V)
{
	for(u32 Index = 0;
	    Index < V.Dimension;
	    ++Index)
	{
		V.Data[Index] *= Scale;
	}
}

inline void
VectorPlusEquals(vec A, vec B)
{
	Assert(A.Dimension == B.Dimension);

	for(u32 Index = 0;
	    Index < A.Dimension;
	    ++Index)
	{
		A.Data[Index] += B.Data[Index];
	}
}

struct matrix
{
	u32 RowCount;
	u32 ColumnCount;
	r32 *Data;
};

inline matrix
MatrixRaw_(memory_pool *Pool, u32 Rows, u32 Columns)
{
	matrix Result = {};
	Result.Data = PoolPushArray(Pool, r32, Rows*Columns);
	Result.ColumnCount = Columns;
	Result.RowCount = Rows;
	return Result;
}

inline matrix
Matrix(memory_pool *Pool, r32 *Data, u32 Rows, u32 Columns)
{
	matrix Result = MatrixRaw_(Pool, Rows, Columns);;

	r32 *ColData = Data;
	r32 *ColDest = Result.Data;
	for(u32 ColumnIndex = 0;
	    ColumnIndex < Result.ColumnCount;
	    ++ColumnIndex)
	{
		r32 *Source = ColData;
		r32 *Dest = ColDest;
		for(u32 RowIndex = 0;
		    RowIndex < Result.RowCount;
		    ++RowIndex)
		{
			*Dest++ = *Source++;
		}

		ColData += Rows;
		ColDest += Result.RowCount;
	}

	return Result;
}

inline matrix
Matrix(r32 *Data, u32 Rows, u32 Columns)
{
	matrix Result = {};
	Result.Data = Data;
	Result.ColumnCount = Columns;
	Result.RowCount = Rows;
	return Result;
}

inline matrix
MatrixRand(memory_pool *Pool, u32 Rows, u32 Columns,
           r32 Mean, r32 StandardDeviation)
{
	matrix Result = MatrixRaw_(Pool, Rows, Columns);

	r32 *ColDest = Result.Data;
	for(u32 ColumnIndex = 0;
	    ColumnIndex < Result.ColumnCount;
	    ++ColumnIndex)
	{
		r32 *Dest = ColDest;
		for(u32 RowIndex = 0;
		    RowIndex < Result.RowCount;
		    ++RowIndex)
		{
			*Dest++ = RandomGaussian(Mean, StandardDeviation);
		}

		ColDest += Result.RowCount;
	}

	return Result;
}

inline vec
Mult(memory_pool *Pool, matrix M, vec V)
{
	Assert(M.ColumnCount == V.Dimension);

	vec Result = VecRaw_(Pool, M.RowCount);

	r32 *Value = Result.Data;
	for(u32 RowIndex = 0;
	    RowIndex < M.RowCount;
	    ++RowIndex)
	{
		r32 *MatValue = M.Data + RowIndex;
		r32 *VValue = V.Data;
	
		*Value = 0.0f;
		for(u32 ColumnIndex = 0;
		    ColumnIndex < M.ColumnCount;
		    ++ColumnIndex)
		{
			*Value += *MatValue * *VValue;
			++VValue;
			MatValue += M.RowCount;
		}

		++Value;
	}

	return Result;
}

inline vec
TransposeMult(memory_pool *Pool, matrix M, vec V)
{
	Assert(M.RowCount == V.Dimension);

	vec Result = VecRaw_(Pool, M.ColumnCount);

	r32 *Value = Result.Data;
	for(u32 ColumnIndex = 0;
	    ColumnIndex < M.ColumnCount;
	    ++ColumnIndex)
	{
		r32 *MatValue = M.Data + ColumnIndex*M.RowCount;
		r32 *VValue = V.Data;

		*Value = 0.0f;
		for(u32 RowIndex = 0;
		    RowIndex < M.RowCount;
		    ++RowIndex)
		{
			*Value += *MatValue++ * *VValue++;
		}

		++Value;
	}

	return Result;	
}

inline matrix
Transpose(memory_pool *Pool, matrix M)
{
	matrix Result = MatrixRaw_(Pool, M.ColumnCount, M.RowCount);

	for(u32 ColumnIndex = 0;
	    ColumnIndex < M.ColumnCount;
	    ++ColumnIndex)
	{
		r32 *Source = M.Data + ColumnIndex*M.RowCount;
		for(u32 RowIndex = 0;
		    RowIndex < M.RowCount;
		    ++RowIndex)
		{
			r32 *Dest = Result.Data + RowIndex*Result.RowCount + ColumnIndex;
			*Dest = *Source++;
		}
	}

	return Result;
}

inline matrix
Mult(memory_pool *Pool, matrix A, matrix B)
{
	Assert(A.ColumnCount == B.RowCount);

	matrix Result = MatrixRaw_(Pool, A.RowCount, B.ColumnCount);

	r32 *Dest = Result.Data;
	for(u32 ColumnIndex = 0;
	    ColumnIndex < Result.ColumnCount;
	    ++ColumnIndex)
	{
		for(u32 RowIndex = 0;
		    RowIndex < Result.RowCount;
		    ++RowIndex)
		{
			*Dest = 0.0f;
			r32 *SourceA = A.Data + RowIndex;
			r32 *SourceB = B.Data + ColumnIndex*B.RowCount;
			for(u32 InnerIndex = 0;
			    InnerIndex < A.ColumnCount;
			    ++InnerIndex)
			{
				*Dest += *SourceA * *SourceB;
				SourceA += A.RowCount;
				++SourceB;
			}
			++Dest;
		}
	}

	return Result;
}

inline matrix
VectorTransposeMult(memory_pool *Pool, vec A, vec B)
{
	matrix Result = MatrixRaw_(Pool, A.Dimension, B.Dimension);

	r32 *Dest = Result.Data;
	for(u32 ColumnIndex = 0;
	    ColumnIndex < Result.ColumnCount;
	    ++ColumnIndex)
	{
		for(u32 RowIndex = 0;
		    RowIndex < Result.RowCount;
		    ++RowIndex)
		{
			*Dest++ = A.Data[RowIndex]*B.Data[ColumnIndex]; 
		}
	}

	return Result;
}

inline void
MatrixPlusEquals(matrix A, matrix B)
{
	Assert(A.RowCount == B.RowCount);
	Assert(A.ColumnCount == B.ColumnCount);

	r32 *AValue = A.Data;
	r32 *BValue = B.Data;
	for(u32 Index = 0;
	    Index < (A.RowCount*A.ColumnCount);
	    ++Index)
	{
		*AValue++ += *BValue++;
	}	
}

inline void
MatrixScaleEquals(r32 Scale, matrix A)
{
	r32 *Dest = A.Data;
	for(u32 Index = 0;
	    Index < (A.RowCount*A.ColumnCount);
	    ++Index)
	{
		*Dest++ *= Scale;
	}
}

inline matrix
MVPlus(memory_pool *Pool, matrix A, vec V)
{
	Assert(V.Dimension == A.RowCount);

	matrix Result = MatrixRaw_(Pool, A.RowCount, A.ColumnCount);

	r32 *Value = Result.Data;
	r32 *MValue = A.Data;
	for(u32 ColumnIndex = 0;
	    ColumnIndex < Result.ColumnCount;
	    ++ColumnIndex)
	{
		r32 *VValue = V.Data;
		for(u32 RowIndex = 0;
		    RowIndex < Result.RowCount;
		    ++RowIndex)
		{
			*Value++ = *MValue++ + *VValue++;
		}
	}

	return Result;
}

inline matrix
Sigmoid(memory_pool *Pool, matrix M)
{
	matrix Result = MatrixRaw_(Pool, M.RowCount, M.ColumnCount);

	r32 *NewValue = Result.Data;
	r32 *OldValue = M.Data;
	for(u32 Index = 0;
	    Index < (Result.RowCount * Result.ColumnCount);
	    ++Index)
	{
		*NewValue++ = Sigmoid(*OldValue++);
	}

	return Result;
}

inline matrix
SigmoidPrime(memory_pool *Pool, matrix M)
{
	matrix Result = MatrixRaw_(Pool, M.RowCount, M.ColumnCount);

	r32 *NewValue = Result.Data;
	r32 *OldValue = M.Data;
	for(u32 Index = 0;
	    Index < (Result.RowCount * Result.ColumnCount);
	    ++Index)
	{
		*NewValue++ = SigmoidPrime(*OldValue++);
	}

	return Result;
}

inline matrix
Minus(memory_pool *Pool, matrix A, matrix B)
{
	Assert((A.RowCount == B.RowCount) && (A.ColumnCount == B.ColumnCount));

	matrix Result = MatrixRaw_(Pool, A.RowCount, A.ColumnCount);

	r32 *AData = A.Data;
	r32 *BData = B.Data;
	r32 *ResultData = Result.Data;
	for(u32 Index = 0;
	    Index < (A.RowCount * A.ColumnCount);
	    ++Index)
	{
		*ResultData++ = *AData++ - *BData++;
	}

	return Result;
}

inline matrix
Hadamard(memory_pool *Pool, matrix A, matrix B)
{
	Assert((A.RowCount == B.RowCount) && (A.ColumnCount == B.ColumnCount));

	matrix Result = MatrixRaw_(Pool, A.RowCount, A.ColumnCount);

	r32 *AData = A.Data;
	r32 *BData = B.Data;
	r32 *ResultData = Result.Data;
	for(u32 Index = 0;
	    Index < (A.RowCount * A.ColumnCount);
	    ++Index)
	{
		*ResultData++ = *AData++ * *BData++;
	}

	return Result;
}

inline matrix
TransposeMult(memory_pool *Pool, matrix A, matrix B)
{
	Assert(A.RowCount == B.RowCount);

	matrix Result = MatrixRaw_(Pool, A.ColumnCount, B.ColumnCount);
	
	r32 *Dest = Result.Data;
	for(u32 ColumnIndex = 0;
	    ColumnIndex < Result.ColumnCount;
	    ++ColumnIndex)
	{
		for(u32 RowIndex = 0;
		    RowIndex < Result.RowCount;
		    ++RowIndex)
		{
			*Dest = 0.0f;
			r32 *SourceA = A.Data + RowIndex*A.RowCount;
			r32 *SourceB = B.Data + ColumnIndex*B.RowCount;
			for(u32 InnerIndex = 0;
			    InnerIndex < A.RowCount;
			    ++InnerIndex)
			{
				*Dest += *SourceA * *SourceB;
				++SourceA;
				++SourceB;
			}
			++Dest;
		}
	}

	return Result;
}

inline matrix
MultTranspose(memory_pool *Pool, matrix A, matrix B)
{
	Assert(A.ColumnCount == B.ColumnCount);

	matrix Result = MatrixRaw_(Pool, A.RowCount, B.RowCount);
	
	r32 *Dest = Result.Data;
	for(u32 ColumnIndex = 0;
	    ColumnIndex < Result.ColumnCount;
	    ++ColumnIndex)
	{
		for(u32 RowIndex = 0;
		    RowIndex < Result.RowCount;
		    ++RowIndex)
		{
			*Dest = 0.0f;
			r32 *SourceA = A.Data + RowIndex;
			r32 *SourceB = B.Data + ColumnIndex;
			for(u32 InnerIndex = 0;
			    InnerIndex < A.ColumnCount;
			    ++InnerIndex)
			{
				*Dest += *SourceA * *SourceB;
				SourceA += A.RowCount;
				SourceB += B.RowCount;
			}
			++Dest;
		}
	}

	return Result;
}

inline vec
MatrixSumColumns(memory_pool *Pool, matrix A)
{
	vec Result = VecZero(Pool, A.RowCount);

	r32 *AData = A.Data;
	for(u32 ColumnIndex = 0;
	    ColumnIndex < A.ColumnCount;
	    ++ColumnIndex)
	{
		r32 *VData = Result.Data;
		for(u32 RowIndex = 0;
		    RowIndex < A.RowCount;
		    ++RowIndex)
		{
			*VData++ += *AData++;
		}
	}

	return Result;
}