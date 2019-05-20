#pragma once

/*
   32-bits Random number generator U[0,1): lfsr113
   Author: Pierre L'Ecuyer,
   Source: http://www.iro.umontreal.ca/~lecuyer/myftp/papers/tausme2.ps
   ---------------------------------------------------------
*/

/* IMPORTANT:
  The initial seeds z1, z2, z3, z4  MUST be larger than
  1, 7, 15, and 127 respectively.
*/

#define DEFAULT_SEED 987654321

struct random
{
	u32 z1,z2,z3,z4;
	b32 ValidSpare;
	r32 SpareGaussian;
};

internal random
SeedRandom(u32 Seed = DEFAULT_SEED)
{
	random Result = {};
	if(Seed <= 127)
	{
		Seed += 127;
	}
	Result.z1 = Seed;
	Result.z2 = Seed;
	Result.z3 = Seed;
	Result.z4 = Seed;

	Result.ValidSpare = false;

	return Result;
}

global_variable random DefaultRandom_ = SeedRandom();
global_variable random *DefaultRandom = &DefaultRandom_;

internal u32
RandomU32(random *Random = DefaultRandom)
{
	// NOTE: lfsr113
    u32 b;
    b  = ((Random->z1 << 6) ^ Random->z1) >> 13;
    Random->z1 = ((Random->z1 & 4294967294U) << 18) ^ b;
    b  = ((Random->z2 << 2) ^ Random->z2) >> 27;
    Random->z2 = ((Random->z2 & 4294967288U) << 2) ^ b;
    b  = ((Random->z3 << 13) ^ Random->z3) >> 21;
    Random->z3 = ((Random->z3 & 4294967280U) << 7) ^ b;
    b  = ((Random->z4 << 3) ^ Random->z4) >> 12;
    Random->z4 = ((Random->z4 & 4294967168U) << 13) ^ b;
    return (Random->z1 ^ Random->z2 ^ Random->z3 ^ Random->z4);
}

inline r32
Random01(random *Random = DefaultRandom)
{
	r32 Result = RandomU32(Random) * 2.3283064365386963e-10f;
	Assert(Result != 1.0f);
	return Result;
}

inline r32
RandomGaussian(r32 Mean, r32 StandardDeviation, random *Random = DefaultRandom)
{
	r32 Result = 0.0f;
	if(Random->ValidSpare)
	{
		Result = Mean + StandardDeviation*Random->SpareGaussian;
		Random->ValidSpare = false;
	}
	else
	{
		r32 u, v, s;
		do
		{
			u = 2.0f*Random01(Random) - 1.0f;
			v = 2.0f*Random01(Random) - 1.0f;
			s = u*u + v*v;
		} while(!(s < 1.0f));

		r32 Scale = SquareRoot((-2.0f * Ln(s))/s);
		Random->SpareGaussian = v*Scale;
		Random->ValidSpare = true;

		Result = (u*Scale)*StandardDeviation + Mean;
	}

	return Result;
}

inline u32
RandomU32InRangeCloseOpen(u32 Lower, u32 Upper, random *Random = DefaultRandom)
{
	u32 Range = Upper - Lower;
	u32 Result = (u32)(Random01(Random)*Range) + Lower;
	return Result;
}

internal void
RandomTest()
{
	random Random = SeedRandom();

	u32 Count[10] = {};

	u32 Trials = 100000;
	for(u32 TrialIndex = 0;
	    TrialIndex < Trials;
	    ++TrialIndex)
	{
		r32 Test = RandomGaussian(0.5f, 0.1f); // Random01(&Random);
		if(Test < 0.1f)
		{
			++Count[0];
		}
		else if(Test < 0.2f)
		{
			++Count[1];
		}
		else if(Test < 0.3f)
		{
			++Count[2];
		}
		else if(Test < 0.4f)
		{
			++Count[3];
		}
		else if(Test < 0.5f)
		{
			++Count[4];
		}
		else if(Test < 0.6f)
		{
			++Count[5];
		}
		else if(Test < 0.7f)
		{
			++Count[6];
		}
		else if(Test < 0.8f)
		{
			++Count[7];
		}
		else if(Test < 0.9f)
		{
			++Count[8];
		}
		else
		{
			++Count[9];
		}
	}

	for(u32 Index = 0;
	    Index < ArrayCount(Count);
	    ++Index)
	{
		u32 Stars = (u32)((10.0f * ArrayCount(Count)) * ((r32)Count[Index] / (r32)Trials));
		for(u32 StarIndex = 0;
		    StarIndex < Stars;
		    ++StarIndex)
		{
			printf("*");
		}
		printf("\n");
	}
}