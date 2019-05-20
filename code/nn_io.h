#pragma once

#define MNIST_OUTPUT_SIZE 10

#pragma pack(push, 1)
struct mnist_image_set_file_header
{
	u32 MagicNumber;
	u32	ImageCount;
	u32 RowCount;
	u32 ColumnCount;
};

struct mnist_label_set_file_header
{
	u32 MagicNumber;
	u32	ItemCount;
};
#pragma pack(pop)

/*
	NOTE: This is the file format for the saved networks. The weight matrices
		and bias vectors have no data for the input layer of neurons.
		
	neural_network_file_header
	layer array
	matrix array
	matrix data
	vector array
	vector data
*/
#define NEURAL_NETWORK_MAGIC_NUMBER 1337
struct neural_network_file_header
{
	u32 MagicNumber;
	
	cost_function CostFn;
	u32 LayerCount;

	u32 LayersOffset;
	u32 WeightMatricesOffset;
	u32 BiasVectorsOffset;
};

struct matrix_serialized
{
	u32 RowCount;
	u32 ColumnCount;
	u32 DataOffset;
};

struct vec_serialized
{
	u32 Dimension;
	u32 DataOffset;
};