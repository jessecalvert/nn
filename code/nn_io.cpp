
internal u32
GetFileSize(FILE *File)
{
	u32 CurrentPosition = ftell(File);
	fseek(File, 0, SEEK_SET);
	u32 BeginningPosition = ftell(File);
	fseek(File, 0, SEEK_END);
	u32 EndPosition = ftell(File);

	u32 Result = EndPosition - BeginningPosition;
	fseek(File, (u32)CurrentPosition, SEEK_SET);
	return Result;
}

internal void
ConvertToLittleEndian(u32 *Value)
{
	*Value = (((*Value & 0xFF) << 24) |
	          ((*Value & 0xFF00) << 8) |
	          ((*Value & 0xFF0000) >> 8) |
	          ((*Value & 0xFF000000) >> 24));
}

internal u8 *
LoadEntireFile(memory_pool *Pool, char *Filename)
{
	u8 *Result = 0;

	FILE *File = 0;
	errno_t Error = fopen_s(&File, Filename, "rb");
	Assert(Error == 0);

	u32 FileSize = GetFileSize(File);
	Result = PoolPushSize(Pool, FileSize);

	umm SizeRead = fread(Result, 1, FileSize, File);
	Assert(SizeRead == FileSize);

	fclose(File);
	return Result;
}

internal mnist_image_set_file_header *
LoadMNISTImageSet(memory_pool *Pool, char *Filename)
{
	u8 *FileData = LoadEntireFile(Pool, Filename);

	ConvertToLittleEndian((u32 *)FileData);
	Assert(*(u32 *)FileData == 2051);

	ConvertToLittleEndian((u32 *)FileData + 1);
	ConvertToLittleEndian((u32 *)FileData + 2);
	ConvertToLittleEndian((u32 *)FileData + 3);

	mnist_image_set_file_header *Result = (mnist_image_set_file_header *)FileData;
	return Result;
}

internal mnist_label_set_file_header *
LoadMNISTLabelSet(memory_pool *Pool, char *Filename)
{
	u8 *FileData = LoadEntireFile(Pool, Filename);

	ConvertToLittleEndian((u32 *)FileData);
	Assert(*(u32 *)FileData == 2049);

	ConvertToLittleEndian((u32 *)FileData + 1);

	mnist_label_set_file_header *Result = (mnist_label_set_file_header *)FileData;
	return Result;
}

internal data_set
LoadMNISTData(memory_pool *Pool, memory_pool *TempPool, char *ImagesFile, char *LabelsFile)
{
	data_set Result = {};

	temp_memory TempMem = PoolBeginTempMemory(TempPool);
	mnist_image_set_file_header *ImagesHeader = LoadMNISTImageSet(TempPool, ImagesFile);
	mnist_label_set_file_header *LabelsHeader = LoadMNISTLabelSet(TempPool, LabelsFile);

	Assert(ImagesHeader->ImageCount == LabelsHeader->ItemCount);
	Result.DataCount = ImagesHeader->ImageCount;

	Result.InputData = PoolPushArray(Pool, vec, Result.DataCount);
	Result.OutputData = PoolPushArray(Pool, vec, Result.DataCount);

	u8 *ImageData = (u8 *)(ImagesHeader + 1);
	u8 *LabelData = (u8 *)(LabelsHeader + 1);

	u32 ImageSize = ImagesHeader->RowCount * ImagesHeader->ColumnCount;
	vec *ImageVec = Result.InputData;
	for(u32 ImageIndex = 0 ;
	    ImageIndex < Result.DataCount;
	    ++ImageIndex)
	{
		*ImageVec = VecRaw_(Pool, ImageSize);
		r32 *Value = ImageVec->Data;
		for(u32 PixelIndex = 0;
		    PixelIndex < ImageSize;
		  	++PixelIndex)
		{
			*Value++ = U8ToR32(*ImageData++);
		}
		++ImageVec;
	}

	vec *LabelVec = Result.OutputData;
	for(u32 LabelIndex = 0 ;
	    LabelIndex < Result.DataCount;
	    ++LabelIndex)
	{
		*LabelVec = VecZero(Pool, MNIST_OUTPUT_SIZE);
		LabelVec->Data[*LabelData++] = 1.0f;
		++LabelVec;
	}

	PoolEndTempMemory(TempMem);

	return Result;
}

internal u32
NetworkGetTotalFileSize(neural_network Network)
{
	u32 Result = 0;
	Result += sizeof(neural_network_file_header);
	Result += Network.LayerCount * sizeof(u32);
	Result += (Network.LayerCount - 1) * sizeof(matrix_serialized);
	Result += (Network.LayerCount - 1) * sizeof(vec_serialized);

	for(u32 LayerIndex = 1;
		LayerIndex < Network.LayerCount;
		++LayerIndex)
	{
		u32 LastLayerSize = Network.Layers[LayerIndex - 1];
		u32 LayerSize = Network.Layers[LayerIndex];

		Result += LastLayerSize * LayerSize * sizeof(r32);
		Result += LayerSize * sizeof(r32);
	}

	return Result;
}

inline u32
OffsetFrom(void *From, void *To)
{
	u32 Result = (u32)((u64)To - (u64)From);
	return Result;
}

internal void
SerializeNetworkToDisk(memory_pool *Pool, neural_network Network, char *Filename)
{
	temp_memory TempMem = PoolBeginTempMemory(Pool);

	FILE *NetworkFile;
	errno_t	Error = fopen_s(&NetworkFile, Filename, "wb");
	Assert(Error == 0);

	u32 TotalFileSize = NetworkGetTotalFileSize(Network);
	neural_network_file_header *Header = (neural_network_file_header *)PoolPushSize(Pool, TotalFileSize);

	Header->MagicNumber = NEURAL_NETWORK_MAGIC_NUMBER;
	Header->CostFn = Network.CostFn;
	Header->LayerCount = Network.LayerCount;

	Header->LayersOffset = sizeof(neural_network_file_header);
	u32 *LayerData = (u32 *)(((u8 *)Header) + Header->LayersOffset);
	for(u32 LayerIndex = 0;
		LayerIndex < Header->LayerCount;
		++LayerIndex)
	{
		*LayerData++ = Network.Layers[LayerIndex];
	}

	Header->WeightMatricesOffset = OffsetFrom(Header, LayerData);
	matrix_serialized *DestMatrix = (matrix_serialized *)(((u8 *)Header) + Header->WeightMatricesOffset);
	r32 *MatrixData = (r32 *)(((u8 *)DestMatrix) + sizeof(matrix_serialized)*(Header->LayerCount - 1));
	for(u32 LayerIndex = 1;
		LayerIndex < Header->LayerCount;
		++LayerIndex)
	{
		matrix *SourceMatrix = Network.WeightMatrices + LayerIndex;
		DestMatrix->RowCount = SourceMatrix->RowCount;
		DestMatrix->ColumnCount = SourceMatrix->ColumnCount;
		DestMatrix->DataOffset = OffsetFrom(Header, MatrixData);
		++DestMatrix;

		r32 *SourceData = SourceMatrix->Data;
		u32 MatrixDataCount = SourceMatrix->RowCount * SourceMatrix->ColumnCount;
		for(u32 MatrixDataIndex = 0;
			MatrixDataIndex < MatrixDataCount;
			++MatrixDataIndex)
		{
			*MatrixData++ = *SourceData++;
		}
	}

	Header->BiasVectorsOffset = OffsetFrom(Header, MatrixData);
	vec_serialized *DestVec = (vec_serialized *)(((u8 *)Header) + Header->BiasVectorsOffset);
	r32 *VecData = (r32 *)(((u8 *)DestVec) + sizeof(vec_serialized)*(Header->LayerCount - 1));
	for(u32 LayerIndex = 1;
		LayerIndex < Header->LayerCount;
		++LayerIndex)
	{
		vec *SourceVec = Network.BiasVectors + LayerIndex;
		DestVec->Dimension = SourceVec->Dimension;
		DestVec->DataOffset = OffsetFrom(Header, VecData);
		++DestVec;

		r32 *SourceData = SourceVec->Data;
		u32 VecDataCount = SourceVec->Dimension;
		for(u32 VecDataIndex = 0;
			VecDataIndex < VecDataCount;
			++VecDataIndex)
		{
			*VecData++ = *SourceData++;
		}
	}

	fwrite(Header, TotalFileSize, 1, NetworkFile);

	fclose(NetworkFile);
	PoolEndTempMemory(TempMem);
}

inline void *
AddOffsetToPointer(void *Pointer, u32 Offset)
{
	void *Result = (u8 *)Pointer + Offset;
	return Result;
}

internal neural_network
LoadNetwork(memory_pool *Pool, char *Filename)
{
	neural_network Result = {};
	neural_network_file_header *Header = (neural_network_file_header *)LoadEntireFile(Pool, Filename);
	Assert(Header->MagicNumber == NEURAL_NETWORK_MAGIC_NUMBER);

	Result.CostFn = Header->CostFn;
	Result.LayerCount = Header->LayerCount;
	
	Result.Layers = (u32 *)AddOffsetToPointer(Header, Header->LayersOffset);	
	Result.WeightMatrices = PoolPushArray(Pool, matrix, Result.LayerCount);
	Result.BiasVectors = PoolPushArray(Pool, vec, Result.LayerCount);

	matrix_serialized *LoadedMatrices = (matrix_serialized *)AddOffsetToPointer(Header, Header->WeightMatricesOffset);
	for(u32 MatrixIndex = 1;
		MatrixIndex < Result.LayerCount;
		++MatrixIndex)
	{
		matrix *Matrix = Result.WeightMatrices + MatrixIndex;
		matrix_serialized *LoadedMatrix = LoadedMatrices + (MatrixIndex - 1);

		Matrix->RowCount = LoadedMatrix->RowCount;
		Matrix->ColumnCount = LoadedMatrix->ColumnCount;
		Matrix->Data = (r32 *)AddOffsetToPointer(Header, LoadedMatrix->DataOffset);
	}

	vec_serialized *LoadedVectors = (vec_serialized *)AddOffsetToPointer(Header, Header->BiasVectorsOffset);
	for(u32 VecIndex = 1;
		VecIndex < Result.LayerCount;
		++VecIndex)
	{
		vec *Vec = Result.BiasVectors + VecIndex;
		vec_serialized *LoadedVec = LoadedVectors + (VecIndex - 1);
		
		Vec->Dimension = LoadedVec->Dimension;
		Vec->Data = (r32 *)AddOffsetToPointer(Header, LoadedVec->DataOffset);
	}

	return Result;
}