extern "C" {

#define G 1.0

__device__ void decode_pair(unsigned long long p, unsigned long long N, 
                            unsigned long long& i, unsigned long long& j) {
    unsigned long long two_N_minus_1 = 2 * N - 1;
    double discriminant = (double)two_N_minus_1 * two_N_minus_1 - 8.0 * (double)p;
    
    if (discriminant < 0) discriminant = 0;
    
    double sqrt_disc = sqrt(discriminant);
    i = (unsigned long long)((two_N_minus_1 - sqrt_disc) / 2.0);
    
    if (i >= N) i = N - 1;
    
    unsigned long long pairs_before = i * (2 * N - i - 1) / 2;
    j = p - pairs_before + i + 1;
    
    if (j >= N) j = N - 1;
    if (j <= i) j = i + 1;
}

__global__ void compute_energy_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    float* __restrict__ result,
    unsigned long long N)
{
    unsigned long long total_pairs = N * (N - 1) / 2;
    unsigned long long p = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float cache[256];
    float local_sum = 0.0;
    
    while (p < total_pairs) {
        unsigned long long i, j;
        decode_pair(p, N, i, j);
        
        float dx = x[i] - x[j];
        float dy = y[i] - y[j];
        float dz = z[i] - z[j];
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        
        if (r > 1e-10f) {
            local_sum += -G / r;
        }
        
        p += gridDim.x * blockDim.x;
    }
    
    cache[threadIdx.x] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            cache[threadIdx.x] += cache[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(result, cache[0]);
    }
}

}