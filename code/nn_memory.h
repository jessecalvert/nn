#pragma once

struct memory_pool
{
	u8 *Base;
	umm TotalSize;
	umm Size;
	u32 TempMemCount;
};

struct temp_memory
{
	memory_pool *Pool;
	umm OldSize;
	u32 TempMemIndex;
};

internal void
PoolInitialize(memory_pool *Pool, u8 *Base, umm TotalSize)
{
	Pool->Base = Base;
	Pool->TotalSize = TotalSize;
	Pool->Size = 0;
}

inline umm
PoolSizeLeft(memory_pool *Pool)
{
	umm Result = Pool->TotalSize - Pool->Size;
	return Result;
}

inline u8*
PoolPushSize(memory_pool *Pool, u32 Size)
{
	u8 *Result = 0;
	Assert(PoolSizeLeft(Pool) >= Size);
	Result = Pool->Base + Pool->Size;
	Pool->Size += Size;
	return Result;
}
#define PoolPushStruct(Pool, type) (type *)PoolPushSize(Pool, sizeof(type))
#define PoolPushArray(Pool, type, Count) (type *)PoolPushSize(Pool, (Count) * sizeof(type))

inline temp_memory
PoolBeginTempMemory(memory_pool *Pool)
{
	temp_memory Result;
	Result.Pool = Pool;
	Result.OldSize = Pool->Size;
	Result.TempMemIndex = ++Pool->TempMemCount;
	return Result;
}

inline void
PoolEndTempMemory(temp_memory TempMem)
{
	memory_pool *Pool = TempMem.Pool;
	Assert(Pool->TempMemCount == TempMem.TempMemIndex)
	Pool->Size = TempMem.OldSize;
	--Pool->TempMemCount;
}

inline void
PoolCheckMemory(memory_pool *Pool)
{
	Assert(Pool->TempMemCount == 0);
}