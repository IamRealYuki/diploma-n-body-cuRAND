#include <stdlib.h>
#include <malloc.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>

struct Point{
    float r, vr, vt;
    float xp, yp, zp;
    float vx, vy, vz;
};

__global__ void gSetupKernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, id, 0, &state[id]);
}

__global__ void gGeneratePoints(curandState* devStates, Point* points, int N){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (id >= N) return;
    
    float r, vr, vt, Z, f0;

    for(;;){
        r = curand_uniform(&devStates[id]);
        vr = -1.0 + 2.0 * curand_uniform(&devStates[id]);
        vt = curand_uniform(&devStates[id]);

        if(1.0 - r*r - vr*vr - vt*vt*(1.0 - r*r) <= 0.0)
            continue;

        f0 = r*r*vt / sqrt(r*r*vt*(1.0 - r*r - vr*vr - vt*vt*(1.0 - r*r)));
        Z = 20.0 * curand_uniform(&devStates[id]);

        if(Z < f0){
            points[id].r = r;
            points[id].vr = vr;
            points[id].vt = vt;
            break;
        }
    }

    float cospsi = -1.0 + 2.0 * curand_uniform(&devStates[id]);
    float phi = 2.0 * M_PI * curand_uniform(&devStates[id]);
    
    float sinpsi = sqrtf(1.0 - cospsi * cospsi);
    float cosphi = cosf(phi);
    float sinphi = sinf(phi);

    points[id].xp = r * sinpsi * cosphi;
    points[id].yp = r * sinpsi * sinphi;
    points[id].zp = r * cospsi;

    points[id].vx = vr * sinpsi * cosphi + vt * cospsi * cosphi - vt * sinphi;
    points[id].vy = vr * sinpsi * sinphi + vt * cospsi * sinphi + vt * cosphi;
    points[id].vz = vr * cospsi - vt * sinpsi;
}

int main(int argc, char* argv[]){
    int N = 1024 * 128;
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) N = 1024 * 128;
    }

    Point* points;
    curandState *devStates;
    float *d_x, *d_y, *d_z, *d_potential;
    float h_potential = 0.0f;

    cudaMalloc((void**)&points, N * sizeof(struct Point));
    cudaMalloc((void**)&devStates, N * sizeof(curandState));
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMalloc((void**)&d_z, N * sizeof(float));
    cudaMalloc((void**)&d_potential, sizeof(float));
    cudaMemset(d_potential, 0, sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    gSetupKernel<<<blocks, threads>>>(devStates);
    cudaDeviceSynchronize();

    gGeneratePoints<<<blocks, threads>>>(devStates, points, N);
    cudaDeviceSynchronize();

    Point* h_points = (Point*)malloc(N * sizeof(struct Point));
    cudaMemcpy(h_points, points, N * sizeof(struct Point), cudaMemcpyDeviceToHost);
    
    float* h_x = (float*)malloc(N * sizeof(float));
    float* h_y = (float*)malloc(N * sizeof(float));
    float* h_z = (float*)malloc(N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        h_x[i] = h_points[i].xp;
        h_y[i] = h_points[i].yp;
        h_z[i] = h_points[i].zp;
    }
    
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice);

    CUmodule module;
    CUfunction kernel;
    
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    CUcontext context;
    cuCtxCreate(&context, 0, device);
    cuModuleLoad(&module, "energy_kernel.cubin");
    cuModuleGetFunction(&kernel, module, "compute_energy_kernel");
    
    void* args[] = { &d_x, &d_y, &d_z, &d_potential, &N };
    
    cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL);
    cuCtxSynchronize();
    
    cuModuleUnload(module);
    cuCtxDestroy(context);
    
    cudaMemcpy(&h_potential, d_potential, sizeof(float), cudaMemcpyDeviceToHost);
    
    float h_kinetic = 0.0f;
    for (int i = 0; i < N; i++) {
        float v2 = h_points[i].vx*h_points[i].vx + 
                   h_points[i].vy*h_points[i].vy + 
                   h_points[i].vz*h_points[i].vz;
        h_kinetic += 0.5f * v2;
    }
    
    float virial_ratio = h_potential / (2.0f * h_kinetic);
    float scale = powf(fabs(virial_ratio), 1.0f/3.0f);
    
    for (int i = 0; i < N; i++) {
        h_points[i].xp *= scale;
        h_points[i].yp *= scale;
        h_points[i].zp *= scale;
        h_points[i].vx *= scale;
        h_points[i].vy *= scale;
        h_points[i].vz *= scale;
    }
    
    for(int i = 0; i < N; i++) {
        printf("%.6f %.6f %.6f %.6f %.6f %.6f\n",
               h_points[i].xp, h_points[i].yp, h_points[i].zp,
               h_points[i].vx, h_points[i].vy, h_points[i].vz);
    }

    free(h_points);
    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(points);
    cudaFree(devStates);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_potential);

    return 0;
}