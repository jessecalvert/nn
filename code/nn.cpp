
#include <stdlib.h>
#include "nn.h"

#include "nn_io.cpp"

internal neural_network
CreateNetwork(memory_pool *Pool, u32 *Layers, u32 LayerCount, cost_function CostFn = CostFn_CrossEntropy)
{
	Assert(LayerCount >= 2);
	
	neural_network Result = {};
	Result.CostFn = CostFn;

	Result.LayerCount = LayerCount;
	Result.Layers = PoolPushArray(Pool, u32, Result.LayerCount);

	u32 *LayerDest = Result.Layers;
	u32 *LayerSource = Layers;
	for(u32 LayerIndex = 0;
	    LayerIndex < Result.LayerCount;
	    ++LayerIndex)
	{
		*LayerDest++ = *LayerSource++;
	}

	Result.WeightMatrices = PoolPushArray(Pool, matrix, Result.LayerCount);
	Result.BiasVectors = PoolPushArray(Pool, vec, Result.LayerCount);

	for(u32 LayerIndex = 1;
	    LayerIndex < Result.LayerCount;
	    ++LayerIndex)
	{
		u32 LayerSize = Result.Layers[LayerIndex];
		u32 LastLayerSize = Result.Layers[LayerIndex - 1];

		r32 WeightStandardDeviation = 1.0f / SquareRoot((r32)LastLayerSize);
		Result.WeightMatrices[LayerIndex] = MatrixRand(Pool, LayerSize, LastLayerSize, 0.0f, WeightStandardDeviation);
		Result.BiasVectors[LayerIndex] = VecRand(Pool, LayerSize, 0.0f, 1.0f);
	}

	return Result;
}

internal feed_forward_result
FeedForward(memory_pool *Pool, neural_network Network, vec Input)
{
	Assert(Input.Dimension == Network.Layers[0]);

	feed_forward_result Result = {};

	Result.Activations = PoolPushArray(Pool, vec, Network.LayerCount);
	Result.WeightedInputs = PoolPushArray(Pool, vec, Network.LayerCount);

	vec *Activations = Result.Activations;
	vec *WeightedInputs = Result.WeightedInputs;

	vec *OldActivation = Activations;
	*Activations++ = Input;
	*WeightedInputs++ = Input;

	for(u32 Index = 1;
	    Index < Network.LayerCount;
	    ++Index)
	{
		matrix *Weight = Network.WeightMatrices + Index;
		vec *Bias = Network.BiasVectors + Index;

		*WeightedInputs = Plus(Pool, Mult(Pool, *Weight, *OldActivation), *Bias);
		*Activations = Sigmoid(Pool, *WeightedInputs);
		OldActivation = Activations;

		++WeightedInputs;
		++Activations;
	}

	return Result;
}

internal feed_forward_batch_result
FeedForwardBatch(memory_pool *Pool, neural_network Network, matrix Inputs)
{
	Assert(Inputs.RowCount == Network.Layers[0]);

	feed_forward_batch_result Result = {};

	Result.Activations = PoolPushArray(Pool, matrix, Network.LayerCount);
	Result.WeightedInputs = PoolPushArray(Pool, matrix, Network.LayerCount);

	matrix *Activations = Result.Activations;
	matrix *WeightedInputs = Result.WeightedInputs;

	matrix *OldActivation = Activations;
	*Activations++ = Inputs;
	*WeightedInputs++ = Inputs;

	for(u32 Index = 1;
	    Index < Network.LayerCount;
	    ++Index)
	{
		matrix *Weight = Network.WeightMatrices + Index;
		vec *Bias = Network.BiasVectors + Index;

		*WeightedInputs = MVPlus(Pool, Mult(Pool, *Weight, *OldActivation), *Bias);
		*Activations = Sigmoid(Pool, *WeightedInputs);
		OldActivation = Activations;

		++WeightedInputs;
		++Activations;
	}

	return Result;
}

internal back_propagate_batch_result
BackPropagateBatch(memory_pool *Pool, neural_network Network,
                   matrix Inputs, matrix DesiredOutputs)
{
	back_propagate_batch_result Result = {};
	feed_forward_batch_result FeedForwardResult = FeedForwardBatch(Pool, Network, Inputs);
	Result.WeightedInputs = FeedForwardResult.WeightedInputs;
	Result.Activations = FeedForwardResult.Activations;

	Result.Errors = PoolPushArray(Pool, matrix, Network.LayerCount);
	matrix *Error = Result.Errors + (Network.LayerCount - 1);

	switch(Network.CostFn)
	{
		case CostFn_Quadratic:
		{
			*Error = Hadamard(Pool, Minus(Pool, FeedForwardResult.Activations[Network.LayerCount - 1], DesiredOutputs),
			                  SigmoidPrime(Pool, FeedForwardResult.WeightedInputs[Network.LayerCount - 1]));
		} break;

		case CostFn_CrossEntropy:
		{
			*Error = Minus(Pool, FeedForwardResult.Activations[Network.LayerCount - 1], DesiredOutputs);
		} break;

		InvalidDefaultCase;
	}

	for(u32 LayerIndex = Network.LayerCount - 2;
	    LayerIndex > 0;
	    --LayerIndex)
	{
		matrix *OldError = Error;
		--Error;

		*Error = Hadamard(Pool, TransposeMult(Pool, Network.WeightMatrices[LayerIndex + 1], *OldError),
		                  SigmoidPrime(Pool, FeedForwardResult.WeightedInputs[LayerIndex]));
	}

	return Result;
}

internal void
GradientDescentBatch(memory_pool *Pool, neural_network Network,
                     matrix Inputs, matrix Outputs,
                     r32 LearningRate, r32 Regularization, u32 TotalTrials)
{
	temp_memory TempMem = PoolBeginTempMemory(Pool);

	back_propagate_batch_result BackPropagateResult = BackPropagateBatch(Pool, Network, Inputs, Outputs);

	u32 TrialCount = Inputs.ColumnCount;
	for(u32 LayerIndex = 1;
	    LayerIndex < Network.LayerCount;
	    ++LayerIndex)
	{
		matrix *Weight = Network.WeightMatrices + LayerIndex;
		vec *Bias = Network.BiasVectors + LayerIndex;

		matrix WeightGradient = MultTranspose(Pool,
            BackPropagateResult.Errors[LayerIndex],
            BackPropagateResult.Activations[LayerIndex - 1]);
		MatrixScaleEquals(-LearningRate/TrialCount, WeightGradient);
		MatrixScaleEquals((1.0f - (LearningRate*Regularization)/TotalTrials), *Weight);
		MatrixPlusEquals(*Weight, WeightGradient);

		vec BiasGradient = MatrixSumColumns(Pool, BackPropagateResult.Errors[LayerIndex]);
		VectorScaleEquals(-LearningRate/TrialCount, BiasGradient);
		VectorPlusEquals(*Bias, BiasGradient);
	}

	PoolEndTempMemory(TempMem);
}

internal void
PrintFeedForwardResult(neural_network Network, feed_forward_result FeedForwardResult)
{
	PrintVec(FeedForwardResult.Activations[Network.LayerCount - 1]);
}

internal void
PrintBackPropagateBatchResult(neural_network Network, back_propagate_batch_result BackPropagateBatchResult)
{
	u32 LayerCount = Network.LayerCount;

	printf("WeightedInputs:\n");
	for(u32 LayerIndex = 0;
	    LayerIndex < LayerCount;
	    ++LayerIndex)
	{
		PrintMatrix(BackPropagateBatchResult.WeightedInputs[LayerIndex]);
	}

	printf("Activations:\n");
	for(u32 LayerIndex = 0;
	    LayerIndex < LayerCount;
	    ++LayerIndex)
	{
		PrintMatrix(BackPropagateBatchResult.Activations[LayerIndex]);
	}

	printf("Errors:\n");
	for(u32 LayerIndex = 1;
	    LayerIndex < LayerCount;
	    ++LayerIndex)
	{
		PrintMatrix(BackPropagateBatchResult.Errors[LayerIndex]);
	}
}

internal batch
CreateMiniBatch(memory_pool *Pool, data_set DataSet, u32 Size)
{
	Assert(Size <= DataSet.DataCount);

	batch Result = {};
	Result.Input = MatrixRaw_(Pool, DataSet.InputData[0].Dimension, Size);
	Result.Output = MatrixRaw_(Pool, DataSet.OutputData[0].Dimension, Size);

	temp_memory TempMem = PoolBeginTempMemory(Pool);
	u32 *Indexes = PoolPushArray(Pool, u32, DataSet.DataCount);

	u32 *IndexAt = Indexes;
	for(u32 Index = 0;
	    Index < DataSet.DataCount;
	    ++Index)
	{
		*IndexAt++ = Index;
	}

	r32 *InputDest = Result.Input.Data;
	r32 *OutputDest = Result.Output.Data;
	for(u32 Index = 0;
	    Index < Size;
	    ++Index)
	{
		u32 NextElementIndex = RandomU32InRangeCloseOpen(Index, DataSet.DataCount);
		u32 NextElement = Indexes[NextElementIndex];
		Indexes[NextElementIndex] = Indexes[Index];

		vec InputVec = DataSet.InputData[NextElement];
		vec OutputVec = DataSet.OutputData[NextElement];

		r32 *InputSource = InputVec.Data;
		for(u32 DataIndex = 0;
		    DataIndex < InputVec.Dimension;
		    ++DataIndex)
		{
			*InputDest++ = *InputSource++;
		}

		r32 *OutputSource = OutputVec.Data;
		for(u32 DataIndex = 0;
		    DataIndex < OutputVec.Dimension;
		    ++DataIndex)
		{
			*OutputDest++ = *OutputSource++;
		}
	}

	PoolEndTempMemory(TempMem);
	return Result;
}

internal batch *
CreateBatches(memory_pool *Pool, data_set DataSet, u32 Size)
{
	Assert((DataSet.DataCount / Size)*Size == DataSet.DataCount);

	u32 BatchCount = DataSet.DataCount / Size;
	batch *Result = PoolPushArray(Pool, batch, BatchCount);

	for(u32 BatchIndex = 0;
	    BatchIndex < BatchCount;
	    ++BatchIndex)
	{
		batch *Batch = Result + BatchIndex;
		Batch->Input = MatrixRaw_(Pool, DataSet.InputData[0].Dimension, Size);
		Batch->Output = MatrixRaw_(Pool, DataSet.OutputData[0].Dimension, Size);
	}

	temp_memory TempMem = PoolBeginTempMemory(Pool);
	u32 *Indexes = PoolPushArray(Pool, u32, DataSet.DataCount);

	u32 *IndexAt = Indexes;
	for(u32 Index = 0;
	    Index < DataSet.DataCount;
	    ++Index)
	{
		*IndexAt++ = Index;
	}

	for(u32 Index = 0;
	    Index < DataSet.DataCount;
	    ++Index)
	{
		u32 NextElementIndex = RandomU32InRangeCloseOpen(Index, DataSet.DataCount);
		u32 NextElement = Indexes[NextElementIndex];
		Indexes[NextElementIndex] = Indexes[Index];
		Indexes[Index] = NextElement;
	}

	r32 *InputDest = 0;
	r32 *OutputDest = 0;
	u32 BatchIndex = 0;
	for(u32 Index = 0;
	    Index < DataSet.DataCount;
	    ++Index)
	{
		u32 NextElement = Indexes[Index];

		if((Index % Size) == 0)
		{
			InputDest = Result[BatchIndex].Input.Data;
			OutputDest = Result[BatchIndex].Output.Data;
			++BatchIndex;
		}
	
		vec InputVec = DataSet.InputData[NextElement];
		vec OutputVec = DataSet.OutputData[NextElement];

		r32 *InputSource = InputVec.Data;
		for(u32 DataIndex = 0;
		    DataIndex < InputVec.Dimension;
		    ++DataIndex)
		{
			*InputDest++ = *InputSource++;
		}

		r32 *OutputSource = OutputVec.Data;
		for(u32 DataIndex = 0;
		    DataIndex < OutputVec.Dimension;
		    ++DataIndex)
		{
			*OutputDest++ = *OutputSource++;
		}
	}

	PoolEndTempMemory(TempMem);
	return Result;
}

internal void
TestNetwork(memory_pool *Pool, neural_network Network, data_set TestSet)
{
	temp_memory TempMem = PoolBeginTempMemory(Pool);

	u32 TotalTrials = TestSet.DataCount;
	u32 Errors = 0;

	batch Batch = CreateMiniBatch(Pool, TestSet, TotalTrials);
	feed_forward_batch_result FeedForward = FeedForwardBatch(Pool, Network, Batch.Input);
	matrix Outputs = FeedForward.Activations[Network.LayerCount - 1];

	for(u32 TrialIndex = 0;
	    TrialIndex < TotalTrials;
	    ++TrialIndex)
	{
		u32 Guess = 123;
		r32 MaxValue = 0.0f;
		r32 *OutputValue = Outputs.Data + TrialIndex*Outputs.RowCount;
		for(u32 OutputIndex = 0;
		    OutputIndex < Outputs.RowCount;
		    ++OutputIndex)
		{
			r32 Value = *OutputValue++;
			if(Value > MaxValue)
			{
				MaxValue = Value;
				Guess = OutputIndex;
			}
		}

		u32 Answer = 124;
		MaxValue = 0.0f;
		r32 *AnswerValue = Batch.Output.Data + TrialIndex*Batch.Output.RowCount;
		for(u32 OutputIndex = 0;
		    OutputIndex < Batch.Output.RowCount;
		    ++OutputIndex)
		{
			r32 Value = *AnswerValue++;
			if(Value > MaxValue)
			{
				MaxValue = Value;
				Answer = OutputIndex;
			}
		}

		Assert((Guess != 123) && (Answer != 124));

		if(Answer != Guess)
		{
			++Errors;
		}
	}

	r32 ErrorRate = (r32)Errors / (r32)TotalTrials;
	r32 SuccessRatePercent = 100.0f*(1 - ErrorRate);

	printf("Success rate: %3.2f%%\n", SuccessRatePercent);

	PoolEndTempMemory(TempMem);
}

internal command_line_options
ParseCommandLineOptions(s32 ArgC, char **ArgV)
{
	command_line_options Result = {};
	Result.HiddenLayerNeurons = 100;
	Result.EpochCount = 0;
	Result.BatchSize = 10;
	Result.LearningRate = 1.0f;
	Result.Regularization = 5.0f;

	for(s32 ArgumentIndex = 1;
		ArgumentIndex < ArgC;
		++ArgumentIndex)
	{
		char *Argument = ArgV[ArgumentIndex];
		if(StringCompare(Argument, "-s"))
		{
			Result.SaveNetwork = ArgV[++ArgumentIndex];
		}
		else if(StringCompare(Argument, "-l"))
		{
			Result.LoadNetwork = ArgV[++ArgumentIndex];
		}
		else if(StringCompare(Argument, "-hiddenlayer"))
		{
			Result.HiddenLayerNeurons = atoi(ArgV[++ArgumentIndex]);
		}
		else if(StringCompare(Argument, "-epochs"))
		{
			Result.EpochCount = atoi(ArgV[++ArgumentIndex]);
		}
		else if(StringCompare(Argument, "-batch"))
		{
			Result.BatchSize = atoi(ArgV[++ArgumentIndex]);
		}
		else if(StringCompare(Argument, "-learningrate"))
		{
			Result.LearningRate = (r32)atof(ArgV[++ArgumentIndex]);
		}
		else if(StringCompare(Argument, "-regularization"))
		{
			Result.Regularization = (r32)atof(ArgV[++ArgumentIndex]);
		}
		else
		{
			InvalidCodePath;
		}
	}

	return Result;
}

s32 main(s32 ArgC, char **ArgV)
{
	command_line_options Options = ParseCommandLineOptions(ArgC, ArgV);

	umm PermanentMemorySize = Gigabytes(1);
	umm TemporaryMemorySize = Megabytes(128);
	umm TotalSize = PermanentMemorySize + TemporaryMemorySize;
	u8 *MainPoolBase = (u8 *)calloc(1, TotalSize);
	u8 *TempPoolBase = MainPoolBase + PermanentMemorySize;
	memory_pool MainPool = {};
	memory_pool TempPool = {};
	PoolInitialize(&MainPool, MainPoolBase, PermanentMemorySize);
	PoolInitialize(&TempPool, TempPoolBase, TemporaryMemorySize);

	data_set TotalTrainingSet = LoadMNISTData(&MainPool, &TempPool, "train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	data_set TrainingSet = TotalTrainingSet;
	TrainingSet.DataCount = 50000;

	data_set VerificationSet = {};
	VerificationSet.DataCount = TotalTrainingSet.DataCount - TrainingSet.DataCount;
	VerificationSet.InputData = TotalTrainingSet.InputData + TrainingSet.DataCount;
	VerificationSet.OutputData = TotalTrainingSet.OutputData + TrainingSet.DataCount;

	data_set TestSet = LoadMNISTData(&MainPool, &TempPool, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	
	neural_network Network = {};
	if(Options.LoadNetwork)
	{
		Network = LoadNetwork(&MainPool, Options.LoadNetwork);
		Assert(TrainingSet.InputData[0].Dimension == Network.Layers[0]);
		Assert(TrainingSet.OutputData[0].Dimension == Network.Layers[2]);
	}
	else
	{
		u32 LayerCount[] =
		{
			TrainingSet.InputData[0].Dimension,
			Options.HiddenLayerNeurons,
			TrainingSet.OutputData[0].Dimension
		};
		Network = CreateNetwork(&MainPool, LayerCount, ArrayCount(LayerCount));
	}

	TestNetwork(&MainPool, Network, TestSet);

	for(u32 EpochIndex = 0;
	    EpochIndex < Options.EpochCount;
	    ++EpochIndex)
	{
		printf("Epoch %d ... ", EpochIndex);
		temp_memory TempMem = PoolBeginTempMemory(&MainPool);
		batch *Batches = CreateBatches(&MainPool, TrainingSet, Options.BatchSize);
		u32 BatchCount = (TrainingSet.DataCount / Options.BatchSize);
		for(u32 BatchIndex = 0;
		    BatchIndex < BatchCount;
		    ++BatchIndex)
		{
			batch *Batch = Batches + BatchIndex;
			GradientDescentBatch(&MainPool, Network, Batch->Input, Batch->Output,
			                     Options.LearningRate, Options.Regularization, TrainingSet.DataCount);
		}

		PoolEndTempMemory(TempMem);
		printf("done\n");
	
		TestNetwork(&MainPool, Network, TestSet);
	}

	if(Options.SaveNetwork)
	{
		SerializeNetworkToDisk(&MainPool, Network, Options.SaveNetwork);
	}

	PoolCheckMemory(&MainPool);
	PoolCheckMemory(&TempPool);
	return 0;
}