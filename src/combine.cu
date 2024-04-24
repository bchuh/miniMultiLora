#include <cuda_runtime.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>

#define MAX_DIMS 10
#define TILE 32
#define BASE_THREAD_NUM 32
#define BLOCK_DIM 1024
typedef float scalar_t;

#define ADD_FUNC       1
#define MUL_FUNC       2
#define ID_FUNC        3
#define NEG_FUNC       4
#define LT_FUNC        5
#define EQ_FUNC        6
#define SIGMOID_FUNC   7
#define RELU_FUNC      8
#define RELU_BACK_FUNC 9
#define LOG_FUNC       10
#define LOG_BACK_FUNC  11
#define EXP_FUNC       12
#define INV_FUNC       13
#define INV_BACK_FUNC  14
#define IS_CLOSE_FUNC  15
#define MAX_FUNC       16
#define POW            17
#define TANH           18

__device__ float fn(int fn_id, float x, float y=0) {
    switch(fn_id) {
      case ADD_FUNC: {
        return x + y;
      }
      case MUL_FUNC: {
        return x * y;
      }
      case ID_FUNC: {
      	return x;
      }
      case NEG_FUNC: {
        return -x;
      }
      case LT_FUNC: {
        if (x < y) {
          return 1.0;
        }
        else {
          return 0.0;
        }
      }
      case EQ_FUNC: {
        if (x == y) {
          return 1.0;
        }
        else {
          return 0.0;
        }
      }
      case SIGMOID_FUNC: {
        if (x >= 0) {
          return 1.0 / (1.0 + exp(-x));
        }
        else {
          return exp(x) / (1.0 + exp(x));
        }
      }
      case RELU_FUNC: {
        return max(x, 0.0);
      }
      case RELU_BACK_FUNC: {
        if (x > 0) {
          return y;
        }
        else {
          return 0.0;
        }
      }
      case LOG_FUNC: {
        return log(x + 1e-6);
      }
      case LOG_BACK_FUNC: {
        return y / (x + 1e-6);
      }
      case EXP_FUNC: {
        return exp(x);
      }
      case INV_FUNC: {
        return float(1.0 / x);
      }
      case INV_BACK_FUNC: {
        return -(1.0 / (x * x)) * y;
      }
      case IS_CLOSE_FUNC: {
        return (x - y < 1e-2) && (y - x < 1e-2);
      }
      case MAX_FUNC: {
        if (x > y) {
          return x;
        }
        else {
          return y;
        }
      }
      case POW: {
	// TODO
        return powf(x, y);
      }
      case TANH: {
	// TODO
        return tanhf(x);
      }
      default: {
        return x + y;
      }
    }
    
}


__device__ int index_to_position(const int* index, const int* strides, int num_dims) {
    int position = 0;
    for (int i = 0; i < num_dims; ++i) {
        position += index[i] * strides[i];
    }
    return position;
}

__device__ void to_index(int ordinal, const int* shape, int* out_index, int num_dims) {
    int cur_ord = ordinal;
    for (int i = num_dims - 1; i >= 0; --i) {
        int sh = shape[i];
        out_index[i] = cur_ord % sh;
        cur_ord /= sh;
    }
}

__device__ void broadcast_index(const int* big_index, const int* big_shape, int num_dims_big, int* out_index, const int* shape, int num_dims) {
  /**
   * Convert a big_index into big_shape to a smaller out_index into shape following broadcasting rules.
   * In this case it may be larger or with more dimensions than the shape given.
   * Additional dimensions may need to be mapped to 0 or removed.
   *
   * Args:
   *    big_index: multidimensional index of bigger tensor
   *    big_shape: tensor shape of bigger tensor
   *    nums_big_dims: number of dimensions in bigger tensor
   *    out_index: multidimensional index of smaller tensor
   *    shape: tensor shape of smaller tensor
   *    num_dims: number of dimensions in smaller tensor
   *
   * Returns:
   *    None (Fills in out_index)
  */
    for (int i = 0; i < num_dims; ++i) {
        if (shape[i] > 1) {
            out_index[i] = big_index[i + (num_dims_big - num_dims)];
        } else {
            out_index[i] = 0;
        }
    }
}
__global__ void SGMVKernel(
  scalar_t* out,
  const int* out_shape,
  const int* out_strides,
  scalar_t* a_storage,
  const int* a_shape,
  const int* a_strides,
  scalar_t* b_storage,
  const int* b_shape,
  const int* b_strides,
  const int* lora_idx_s
) {
  // general SGMV kernal, call twice with different a and b can work as expand or shrink kernal
  int n_lora = blockDim.z;
  int batch_size_offset = lora_idx_s[blockIdx.z];
  int row_limit[2]; //dim order: (row, col). row_limit = {lower_bound_inclusive, upper_bound_exclusive}
  // this is the row idx limit of the output matrix
  row_limit[0] = lora_idx_s[blockIdx.z]; // inclusive. lora_idx_s[blockIdx.z] is the start of current group
  row_limit[1] = lora_idx_s[blockIdx.z+1]; // exclusive



  ////////////////// mat mult: a*b /////////////////////


  __shared__ scalar_t a_shared[TILE][TILE];
  __shared__ scalar_t b_shared[TILE][TILE];

  // In each block, we will compute a batch of the output matrix
  // All the threads in the block will work together to compute this batch
  int batch = blockIdx.z;



  /// BEGIN ASSIGN1_2
  /// TODO
  // Hints:
  int temp_pos_2d[2];
  int temp_pos_3d[3];
  //int int_index[MAX_DIMS];

  // 1. Compute the row and column of the output matrix this block will compute
  int i = batch_size_offset + blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  // 2. Compute the position in the output array that this thread will write to

  int N = a_shape[2]; //N here means m, the second dim of input x

  // 3. Iterate over tiles of the two input matrices, read the data into shared memory
  int tile_i = threadIdx.x;
  int tile_j = threadIdx.y;
  double out_temp = 0;
  for(int ks = 0; ks < N; ks+=TILE){
      temp_pos_2d[0] = i;
      temp_pos_2d[1] = ks+tile_j;
      if(i<row_limit[1] && otemp_pos_2d[1] <N){
          a_shared[tile_i][tile_j] = a_storage[index_to_position(temp_pos_2d, a_strides, 2)];
      }else
          a_shared[tile_i][tile_j] = 0;
      temp_pos_3d[0] = blockIdx.z;
      temp_pos_3d[1] = ks+tile_i;
      temp_pos_3d[2] = j;
      if (temp_pos_3d[1] < N && j < b_shape[2]){
          b_shared[tile_i][tile_j] = b_storage[index_to_position(temp_pos_3d, b_strides, 3)];
      }else
          b_shared[tile_i][tile_j] = 0;

      __syncthreads();
      for(int ki = 0; ki < TILE; ki++){
          out_temp += a_shared[tile_i][ki] * b_shared[ki][tile_j];
      }

      __syncthreads();
  }
  if (i<row_limit[1] && j<out_shape[1]){
      temp_pos_2d[0] = i;
      temp_pos_2d[1] = j;
      out[index_to_position(temp_pos_2d, out_strides, 2)] = out_temp;

  }
}

__global__ void MatrixMultiplyKernel(
    scalar_t* out,
    const int* out_shape,
    const int* out_strides,
    scalar_t* a_storage,
    const int* a_shape,
    const int* a_strides,
    scalar_t* b_storage,
    const int* b_shape,
    const int* b_strides
) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix. Matrix a and b are both in a batch
   * format, with shape [batch_size, m, n], [batch_size, n, p].
   * Requirements:
   * - All data must be first moved to shared memory.
   * - Only read each cell in a and b once.
   * - Only write to global memory once per kernel.
   * There is guarantee that a_shape[0] == b_shape[0], a_shape[2] == b_shape[1],
   * and out_shape[0] == a_shape[0], out_shape[1] == b_shape[1]
   *
   * Args:
   *   out: compact 1D array of size batch_size x m x p to write the output to
   *   out_shape: shape of the output array
   *   out_strides: strides of the output array
   *   a_storage: compact 1D array of size batch_size x m x n
   *   a_shape: shape of the a array
   *   a_strides: strides of the a array
   *   b_storage: comapct 2D array of size batch_size x n x p
   *   b_shape: shape of the b array
   *   b_strides: strides of the b array
   *
   * Returns:
   *   None (Fills in out array)
   */


    __shared__ scalar_t a_shared[TILE][TILE];
    __shared__ scalar_t b_shared[TILE][TILE];

    // In each block, we will compute a batch of the output matrix
    // All the threads in the block will work together to compute this batch
    int batch = blockIdx.z;
    int a_batch_stride = a_shape[0] > 1 ? a_strides[0] : 0;
    int b_batch_stride = b_shape[0] > 1 ? b_strides[0] : 0;


    /// BEGIN ASSIGN1_2
    /// TODO
    // Hints:
    int out_index[MAX_DIMS];
    //int int_index[MAX_DIMS];
    int out_size = 1;
    for(int i=0; i<3; ++i){
        if(out_shape[i]!=0)
            out_size*=out_shape[i];
    }
    // 1. Compute the row and column of the output matrix this block will compute
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // 2. Compute the position in the output array that this thread will write to
    out_index[0] = batch;
    out_index[1] = i;
    out_index[2] = j;

    int N = a_shape[2];

    // 3. Iterate over tiles of the two input matrices, read the data into shared memory
    int tile_i = threadIdx.x;
    int tile_j = threadIdx.y;
    double out_temp = 0;
    for(int ks = 0; ks < N; ks+=TILE){
        out_index[0] = batch;
        out_index[1] = i;
        out_index[2] = ks+tile_j;
        if(i<a_shape[1] && out_index[2] <N){
            a_shared[tile_i][tile_j] = a_storage[index_to_position(out_index, a_strides, 3)];
        }else
            a_shared[tile_i][tile_j] = 0;
        out_index[1] = ks+tile_i;
        out_index[2] = j;
        if (out_index[1] < N && j < b_shape[2]){
            b_shared[tile_i][tile_j] = b_storage[index_to_position(out_index, b_strides, 3)];
        }else
            b_shared[tile_i][tile_j] = 0;

        __syncthreads();
        for(int ki = 0; ki < TILE; ki++){

            if(i==32 && j==0 && tile_i ==0&&ki==0)
                // printf("a: %f , b: %f \n", a_shared[tile_i][ki], b_shared[ki][tile_j]);
            out_temp += a_shared[tile_i][ki] * b_shared[ki][tile_j];
        }

        __syncthreads();
        if(i==32 && j==0)
          // printf("%f \n", out_temp);
    }
    if (i<out_shape[1] && j<out_shape[2]){
        out_index[1] = i;
        out_index[2] = j;
        out[index_to_position(out_index, out_strides, 3)] = out_temp;//out_temp;

    }

    /// END ASSIGN1_2
}


__global__ void mapKernel(
    scalar_t* out,
    int* out_shape,
    int* out_strides,
    int out_size,
    scalar_t* in_storage,
    int* in_shape,
    int* in_strides,
    int shape_size,
    int fn_id
) {
  /**
   * Map function. Apply a unary function to each element of the input array and store the result in the output array.
   * Optimization: Parallelize over the elements of the output array.
   *
   * You may find the following functions useful:
   * - index_to_position: converts an index to a position in a compact array
   * - to_index: converts a position to an index in a multidimensional array
   * - broadcast_index: converts an index in a smaller array to an index in a larger array
   *
   * Args:
   *  out: compact 1D array of size out_size to write the output to
   *  out_shape: shape of the output array
   *  out_strides: strides of the output array
   *  out_size: size of the output array
   *  in_storage: compact 1D array of size in_size
   *  in_shape: shape of the input array
   *  in_strides: strides of the input array
   *  shape_size: number of dimensions in the input and output arrays, assume dimensions are the same
   *  fn_id: id of the function to apply to each element of the input array
   *
   * Returns:
   *  None (Fills in out array)
   */
  int out_index[MAX_DIMS];
  int in_index[MAX_DIMS];

  /// BEGIN ASSIGN1_2
  /// TODO
  // Hints:
  // 1. Compute the position in the output array that this thread will write to
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if(pos>=out_size) return;
  // 2. Convert the position to the out_index according to out_shape
  to_index(pos, out_shape, out_index, shape_size);
  // 3. Broadcast the out_index to the in_index according to in_shape (optional in some cases)
  broadcast_index(out_index, out_shape, shape_size, in_index, in_shape, shape_size);
  //  broadcast_index(const int* big_index, const int* big_shape, int num_dims_big, int* out_index, const int* shape, int num_dims)
  // 4. Calculate the position of element in in_array according to in_index and in_strides
  int in_pos = index_to_position(in_index, in_strides, shape_size);
  // 5. Calculate the position of element in out_array according to out_index and out_strides

  // 6. Apply the unary function to the input element and write the output to the out memory
  out[pos] = fn(fn_id, in_storage[in_pos]);
  // assert(false && "Not Implemented");
  /// END ASSIGN1_2
  /// END ASSIGN1_2
}


__global__ void reduceKernel(
    scalar_t* out,
    int* out_shape,
    int* out_strides,
    int out_size,
    scalar_t* a_storage,
    int* a_shape,
    int* a_strides,
    int reduce_dim,
    double reduce_value,
    int shape_size,
    int fn_id
) {
  /**
   * Reduce function. Apply a reduce function to elements of the input array a and store the result in the output array.
   * Optimization:
   * Parallelize over the reduction operation. Each kernel performs one reduction.
   * e.g. a = [[1, 2, 3], [4, 5, 6]], kernel0 computes reduce([1, 2, 3]), kernel1 computes reduce([4, 5, 6]).
   *
   * You may find the following functions useful:
   * - index_to_position: converts an index to a position in a compact array
   * - to_index: converts a position to an index in a multidimensional array
   *
   * Args:
   *  out: compact 1D array of size out_size to write the output to
   *  out_shape: shape of the output array
   *  out_strides: strides of the output array
   *  out_size: size of the output array
   *  a_storage: compact 1D array of size in_size
   *  a_shape: shape of the input array
   *  a_strides: strides of the input array
   *  reduce_dim: dimension to reduce on
   *  reduce_value: initial value for the reduction
   *  shape_size: number of dimensions in the input & output array, assert dimensions are the same
   *  fn_id: id of the reduce function, currently only support add, multiply, and max
   *
   *
   * Returns:
   *  None (Fills in out array)
   */

    // __shared__ double cache[BLOCK_DIM]; // Uncomment this line if you want to use shared memory to store partial results
    int out_index[MAX_DIMS];

    /// BEGIN ASSIGN1_2
    /// TODO
    // 1. Define the position of the output element that this thread or this block will write to
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    // 2. Convert the out_pos to the out_index according to out_shape
    if(pos>=out_size) return;
    to_index(pos, out_shape, out_index, shape_size);
    // 3. Initialize the reduce_value to the output element
    out[pos] = reduce_value;
    // 4. Iterate over the reduce_dim dimension of the input array to compute the reduced value
    for(int i=0; i<a_shape[reduce_dim]; ++i){
        out_index[reduce_dim] = i;
        int a_pos = index_to_position(out_index, a_strides, shape_size);
        out[pos] = fn(fn_id, out[pos], a_storage[a_pos]);
    }
    /// END ASSIGN1_2
}

__global__ void zipKernel(
    scalar_t* out,
    int* out_shape,
    int* out_strides,
    int out_size,
    int out_shape_size,
    scalar_t* a_storage,
    int* a_shape,
    int* a_strides,
    int a_shape_size,
    scalar_t* b_storage,
    int* b_shape,
    int* b_strides,
    int b_shape_size,
    int fn_id
) {
  /**
   * Zip function. Apply a binary function to elements of the input array a & b and store the result in the output array.
   * Optimization: Parallelize over the elements of the output array.
   *
   * You may find the following functions useful:
   * - index_to_position: converts an index to a position in a compact array
   * - to_index: converts a position to an index in a multidimensional array
   * - broadcast_index: converts an index in a smaller array to an index in a larger array
   *
   * Args:
   *  out: compact 1D array of size out_size to write the output to
   *  out_shape: shape of the output array
   *  out_strides: strides of the output array
   *  out_size: size of the output array
   *  out_shape_size: number of dimensions in the output array
   *  a_storage: compact 1D array of size in_size
   *  a_shape: shape of the input array
   *  a_strides: strides of the input array
   *  a_shape_size: number of dimensions in the input array
   *  b_storage: compact 1D array of size in_size
   *  b_shape: shape of the input array
   *  b_strides: strides of the input array
   *  b_shape_size: number of dimensions in the input array
   *  fn_id: id of the function to apply to each element of the a & b array
   *
   *
   * Returns:
   *  None (Fills in out array)
   */

    int out_index[MAX_DIMS];
    int a_index[MAX_DIMS];
    int b_index[MAX_DIMS];

    /// BEGIN ASSIGN1_2
    /// TODO
    // Hints:
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if(pos>=out_size) return;
    // 2. Convert the position to the out_index according to out_shape
    to_index(pos, out_shape, out_index, out_shape_size);
    // 3. Calculate the position of element in out_array according to out_index and out_strides
    // 4. Broadcast the out_index to the a_index according to a_shape
    broadcast_index(out_index, out_shape, out_shape_size, a_index, a_shape, a_shape_size);
    // 5. Calculate the position of element in a_array according to a_index and a_strides
    int a_pos = index_to_position(a_index, a_strides, a_shape_size);
    // 6. Broadcast the out_index to the b_index according to b_shape
    broadcast_index(out_index, out_shape, out_shape_size, b_index, b_shape, b_shape_size);
    // 7.Calculate the position of element in b_array according to b_index and b_strides
    int b_pos = index_to_position(b_index, b_strides, b_shape_size);
    // 8. Apply the binary function to the input elements in a_array & b_array and write the output to the out memory
    out[pos] = fn(fn_id, a_storage[a_pos], b_storage[b_pos]); // , b_storage[b_pos]

    /// END ASSIGN1_2
}

/*

__global__ void MatrixMultiplyKernel(
    float* out,
    const int* out_shape,
    const int* out_strides,
    float* a_storage,
    const int* a_shape,
    const int* a_strides,
    float* b_storage,
    const int* b_shape,
    const int* b_strides
) {

    assert(false && "Not Implemented");
}


__global__ void mapKernel(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    float* in_storage, 
    int* in_shape, 
    int* in_strides,
    int shape_size,
    int fn_id
) {
    assert(false && "Not Implemented");
}


__global__ void reduceKernel(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    float* a_storage, 
    int* a_shape, 
    int* a_strides, 
    int reduce_dim,
    float reduce_value,
    int shape_size,
    int fn_id
) {
    assert(false && "Not Implemented");
}

__global__ void zipKernel(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size,
    int out_shape_size,
    float* a_storage, 
    int* a_shape, 
    int* a_strides,
    int a_shape_size,
    float* b_storage, 
    int* b_shape, 
    int* b_strides,
    int b_shape_size,
    int fn_id
) {
    assert(false && "Not Implemented");
}
 */


extern "C" {
  int maxLoraGroupSize(int* lora_idx_s, int length) {
    if (lora_idx_s == NULL || length <= 1) {
        // No max distance can be calculated with fewer than 2 elements
        printf("## error: lora_idx_s <= 1")
        return -1;
    }

    int maxDist = 0;

    for (int i = 1; i < length; i++) {
        // Calculate the distance between the current and previous index
        int currentDist = lora_idx_s[i] - lora_idx_s[i - 1];

        // Update maxDist if the current distance is larger
        if (currentDist > maxDist) {
            maxDist = currentDist;
        }
    }

    return maxDist;
}
  void launchSGMV(
    float* in_storage,
    int* in_shape,
    int* in_strides,
    float* out,
    int* out_shape,
    int* out_strides,
    float* a_storage,
    int* a_shape,
    int* a_strides,
    float* b_storage,
    int* b_shape,
    int* b_strides,
    int* lora_idx_s,
    int m, int p 
) {
    // in matrix should be batch_size * m
    // a and b (lora matrixs) should be n_lora * m * n and n_lora * n * p
    // m and p here means the in_dim and out_dim of actural linear weight matrix
    // lora_idx_s: array of idxs, 0<=lora_idx_s[i]<batch_size, represent the start idx of each lora input group.
    // e.g., input 0~7: lora A, input 8~10: lora B. lora_idx_s: [0, 7, 10]
    // m: input hidden_dim
    int batch = in_shape[0];
    int n_lora = a_shape[0];
    int n = a_shape[2];
    // max_lora_group_size: the size (number of tokens) of the largest lora group in input matrix 
    int max_lora_group_size = maxLoraGroupSize(lora_idx_s, n_lora+1);
    // n means lora rank (low rank space)
    // Allocate device memory
    float *d_out, *d_a, *d_v, *d_b, *d_in;
    cudaMalloc(&d_in, batch * m * sizeof(float));
    cudaMalloc(&d_a, n_lora * m * n * sizeof(float));
    cudaMalloc(&d_b, n_lora * n * p * sizeof(float));
    cudaMalloc(&d_v, batch * n * sizeof(float));
    cudaMalloc(&d_out, batch * p * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides, *d_b_shape, *d_b_strides, *d_in_shape, *d_in_strides, *d_lora_idx_s;
    cudaMalloc(&d_in_shape, 2 * sizeof(int));
    cudaMalloc(&d_in_strides, 2 * sizeof(int));
    cudaMalloc(&d_out_shape, 2 * sizeof(int));
    cudaMalloc(&d_out_strides, 2 * sizeof(int));
    cudaMalloc(&d_a_shape, 3 * sizeof(int));
    cudaMalloc(&d_a_strides, 3 * sizeof(int));
    cudaMalloc(&d_b_shape, 3 * sizeof(int)); // leave a, b as 3d because first dim is n_lora
    cudaMalloc(&d_b_strides, 3 * sizeof(int));
    cudaMalloc(&d_lora_idx_s, (n_lora+1) * sizeof(int)); //+1 because the first element is 0
    cudaMalloc(&d_v_shape, 2 * sizeof(int));
    cudaMalloc(&d_v_strides, 2 * sizeof(int));
    int v_shape[2] = {batch, n};
    int v_strides[2] = {n, 1};
    // Copy data to the device
    cudaMemcpy(d_in, in_storage, batch * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, a_storage, n_lora * m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_storage, n_lora * n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_shape, in_shape, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_strides, in_strides, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lora_idx_s, lora_idx_s, (n_lora+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_shape, &v_shape, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_strides, &v_strides, 2 * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = BASE_THREAD_NUM;
    dim3 blockDims(threadsPerBlock, threadsPerBlock, 1); // Adjust these values based on your specific requirements
    dim3 gridDims((max_lora_group_size + threadsPerBlock - 1) / threadsPerBlock, (n + threadsPerBlock - 1) / threadsPerBlock, n_lora);
    SGMVKernel<<<gridDims, blockDims>>>(
        d_v, d_v_shape, d_v_strides, d_in, d_in_shape, d_in_strides, d_a, d_a_shape, d_a_strides, d_lora_idx_s
    );
    cudaDeviceSynchronize();
    dim3 gridDims2((max_lora_group_size + threadsPerBlock - 1) / threadsPerBlock, (p + threadsPerBlock - 1) / threadsPerBlock, n_lora);
    
    SGMVKernel<<<gridDims2, blockDims>>>(
        d_out, d_out_shape, d_out_strides, d_v, d_v_shape, d_v_strides, d_b, d_b_shape, d_b_strides, d_lora_idx_s
    );

    // Copy back to the host
    cudaMemcpy(out, d_out, batch * m * p * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Matmul Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_shape);
    cudaFree(d_b_strides);
    cudaFree(d_in);
    cudaFree(d_v);
    cudaFree(d_in_shape);
    cudaFree(d_in_strides);
    cudaFree(d_lora_idx_s);
    cudaFree(d_v_shape);
    cudaFree(d_v_strides);
}

void MatrixMultiply(
    float* out,
    int* out_shape,
    int* out_strides,
    float* a_storage,
    int* a_shape,
    int* a_strides,
    float* b_storage,
    int* b_shape,
    int* b_strides,
    int batch, int m, int p
) {
    int n = a_shape[2];

    // Allocate device memory
    float *d_out, *d_a, *d_b;
    cudaMalloc(&d_a, batch * m * n * sizeof(float));
    cudaMalloc(&d_b, batch * n * p * sizeof(float));
    cudaMalloc(&d_out, batch * m * p * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides, *d_b_shape, *d_b_strides;
    cudaMalloc(&d_out_shape, 3 * sizeof(int));
    cudaMalloc(&d_out_strides, 3 * sizeof(int));
    cudaMalloc(&d_a_shape, 3 * sizeof(int));
    cudaMalloc(&d_a_strides, 3 * sizeof(int));
    cudaMalloc(&d_b_shape, 3 * sizeof(int));
    cudaMalloc(&d_b_strides, 3 * sizeof(int));


    // Copy data to the device
    cudaMemcpy(d_a, a_storage, batch * m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_storage, batch * n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = BASE_THREAD_NUM;
    dim3 blockDims(threadsPerBlock, threadsPerBlock, 1); // Adjust these values based on your specific requirements
    dim3 gridDims((m + threadsPerBlock - 1) / threadsPerBlock, (p + threadsPerBlock - 1) / threadsPerBlock, batch);
    MatrixMultiplyKernel<<<gridDims, blockDims>>>(
        d_out, d_out_shape, d_out_strides, d_a, d_a_shape, d_a_strides, d_b, d_b_shape, d_b_strides
    );

    // Copy back to the host
    cudaMemcpy(out, d_out, batch * m * p * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Matmul Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_shape);
    cudaFree(d_b_strides);
}

void tensorMap(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    float* in_storage, 
    int* in_shape, 
    int* in_strides,
    int in_size,
    int shape_size,
    int fn_id
) {

    float *d_out, *d_in;
    cudaMalloc(&d_out, out_size * sizeof(float));
    cudaMalloc(&d_in, in_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_in_shape, *d_in_strides;
    cudaMalloc(&d_out_shape, shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, shape_size * sizeof(int));
    cudaMalloc(&d_in_shape, shape_size * sizeof(int));
    cudaMalloc(&d_in_strides, shape_size * sizeof(int));

    cudaMemcpy(d_in, in_storage, in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_shape, in_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_strides, in_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    mapKernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_out, d_out_shape, d_out_strides, out_size, 
      d_in, d_in_shape, d_in_strides, 
      shape_size, fn_id);
    
    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Map Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_in_shape);
    cudaFree(d_in_strides);
}


void tensorZip(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size,
    int out_shape_size,
    float* a_storage, 
    int* a_shape, 
    int* a_strides,
    int a_size,
    int a_shape_size,
    float* b_storage, 
    int* b_shape, 
    int* b_strides,
    int b_size,
    int b_shape_size,
    int fn_id
) {

    // Allocate device memory
    float *d_out, *d_a, *d_b;
    cudaMalloc((void **)&d_a, a_size * sizeof(float));
    cudaMalloc(&d_b, b_size * sizeof(float));
    cudaMalloc(&d_out, out_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides, *d_b_shape, *d_b_strides;
    cudaMalloc(&d_out_shape, out_shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, out_shape_size * sizeof(int));
    cudaMalloc(&d_a_shape, a_shape_size * sizeof(int));
    cudaMalloc(&d_a_strides, a_shape_size * sizeof(int));
    cudaMalloc(&d_b_shape, b_shape_size * sizeof(int));
    cudaMalloc(&d_b_strides, b_shape_size * sizeof(int));

    // Copy data to the device
    cudaMemcpy(d_a, a_storage, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_storage, b_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, out_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, out_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, a_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, a_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b_shape, b_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b_strides, b_shape_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    zipKernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_out, d_out_shape, d_out_strides, out_size, out_shape_size,
      d_a, d_a_shape, d_a_strides, a_shape_size,
      d_b, d_b_shape, d_b_strides, b_shape_size,
      fn_id);

    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();


    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Zip Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_shape);
    cudaFree(d_b_strides);
}



void tensorReduce(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    float* a_storage, 
    int* a_shape, 
    int* a_strides, 
    int reduce_dim, 
    float reduce_value,
    int shape_size,
    int fn_id
) {
    int a_size = out_size * a_shape[reduce_dim];
    float *d_out, *d_a;
    cudaMalloc(&d_out, out_size * sizeof(float));
    cudaMalloc(&d_a, a_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides;
    cudaMalloc(&d_out_shape, shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, shape_size * sizeof(int));
    cudaMalloc(&d_a_shape, shape_size * sizeof(int));
    cudaMalloc(&d_a_strides, shape_size * sizeof(int));

    cudaMemcpy(d_a, a_storage, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_out, d_out_shape, d_out_strides, out_size, 
        d_a, d_a_shape, d_a_strides, 
        reduce_dim, reduce_value, shape_size, fn_id
    );

    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Reduce Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }

    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
}

}