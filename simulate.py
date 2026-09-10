import pycuda.driver as drv
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import sys
import numpy as np
import octree_lib

CUDA_KERNEL = """
struct GPUNode {
    float com_x, com_y, com_z;
    float size;
    float mass;
    float center_x, center_y, center_z;
    int child_index;
    int is_leaf;
    int is_empty;
};

__global__ void compute_acceleration_bh_kernel(
    float* pos, float* acc, GPUNode* nodes,
    int num_nodes, int N, float theta, float soft, float G
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    float px = pos[i * 3 + 0];
    float py = pos[i * 3 + 1];
    float pz = pos[i * 3 + 2];
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        if (node_idx >= num_nodes) continue;
        
        GPUNode node = nodes[node_idx];
        if (node.is_empty) continue;
        
        float dx = node.com_x - px;
        float dy = node.com_y - py;
        float dz = node.com_z - pz;
        float dist2 = dx*dx + dy*dy + dz*dz + 1e-10f;
        float dist = sqrtf(dist2);
        
        if (node.is_leaf || (node.size / dist) < theta) {
            if (dist > 1e-10f) {
                float inv_r3 = 1.0f / (dist2 * dist + soft * soft * dist);
                float force = G * node.mass * inv_r3;
                ax += force * dx;
                ay += force * dy;
                az += force * dz;
            }
        } else {
            if (!node.is_leaf && node.child_index >= 0) {
                for (int j = 0; j < 8; j++) {
                    int child_idx = node.child_index + j;
                    if (child_idx < num_nodes && !nodes[child_idx].is_empty)
                        stack[stack_ptr++] = child_idx;
                }
            }
        }
    }
    acc[i * 3 + 0] = ax;
    acc[i * 3 + 1] = ay;
    acc[i * 3 + 2] = az;
}

__global__ void update_positions_kernel(
    float* pos, float* vel, float* acc, float dt, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int idx = i * 3;
    float half_dt2 = 0.5f * dt * dt;
    pos[idx]   += vel[idx] * dt + acc[idx] * half_dt2;
    pos[idx+1] += vel[idx+1] * dt + acc[idx+1] * half_dt2;
    pos[idx+2] += vel[idx+2] * dt + acc[idx+2] * half_dt2;
}

__global__ void update_velocities_kernel(
    float* vel, float* acc, float* new_acc, float dt, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int idx = i * 3;
    float half_dt = 0.5f * dt;
    vel[idx]   += (acc[idx] + new_acc[idx]) * half_dt;
    vel[idx+1] += (acc[idx+1] + new_acc[idx+1]) * half_dt;
    vel[idx+2] += (acc[idx+2] + new_acc[idx+2]) * half_dt;
}
"""

class NBodySimulationGPU:
    def __init__(self, theta=0.7, rebuild_interval=5):
        self.dt = 0.01
        self.softening = 0.01
        self.G = 1.0
        self.theta = theta
        self.rebuild_interval = rebuild_interval
        self.step_counter = 0
        self.N = 0
        self.pos = None
        self.vel = None
        self.masses = None
        
        self.module = SourceModule(CUDA_KERNEL, options=['-use_fast_math'])
        self.accel_kernel = self.module.get_function('compute_acceleration_bh_kernel')
        self.update_pos_kernel = self.module.get_function('update_positions_kernel')
        self.update_vel_kernel = self.module.get_function('update_velocities_kernel')
    
    def set_data(self, positions, velocities):
        self.N = positions.shape[0]
        self.pos = np.ascontiguousarray(positions, dtype=np.float32)
        self.vel = np.ascontiguousarray(velocities, dtype=np.float32)
        self.masses = np.ones(self.N, dtype=np.float32)
        
        self.pos_gpu = gpuarray.to_gpu(self.pos.ravel())
        self.vel_gpu = gpuarray.to_gpu(self.vel.ravel())
        self.acc_gpu = gpuarray.zeros(self.N * 3, dtype=np.float32)
        self.new_acc_gpu = gpuarray.zeros(self.N * 3, dtype=np.float32)
        self.masses_gpu = gpuarray.to_gpu(self.masses)
        self.rebuild_tree()
    
    def rebuild_tree(self):
        pos_flat = self.pos.ravel()
        gpu_data = octree_lib.build_octree_gpu(
            pos_flat[0::3].astype(np.float32),
            pos_flat[1::3].astype(np.float32),
            pos_flat[2::3].astype(np.float32),
            self.masses
        )
        num_nodes = gpu_data.shape[0]
        gpu_nodes = np.empty(num_nodes, dtype=np.dtype([
            ('com_x', np.float32), ('com_y', np.float32), ('com_z', np.float32),
            ('size', np.float32), ('mass', np.float32),
            ('center_x', np.float32), ('center_y', np.float32), ('center_z', np.float32),
            ('child_index', np.int32), ('is_leaf', np.int32), ('is_empty', np.int32)
        ]))
        for i, f in enumerate(['com_x', 'com_y', 'com_z', 'size', 'mass', 'center_x', 'center_y', 'center_z']):
            gpu_nodes[f] = gpu_data[:, i]
        gpu_nodes['child_index'] = gpu_data[:, 8].astype(np.int32)
        gpu_nodes['is_leaf'] = gpu_data[:, 9].astype(np.int32)
        gpu_nodes['is_empty'] = gpu_data[:, 10].astype(np.int32)
        self.nodes_gpu = gpuarray.to_gpu(gpu_nodes)
        self.num_nodes_gpu = num_nodes
    
    def step(self):
        self.step_counter += 1
        if (self.step_counter % self.rebuild_interval == 0) or (self.step_counter == 1):
            self.pos = self.pos_gpu.get().reshape(self.N, 3)
            self.rebuild_tree()
        
        threads = 256
        blocks = (self.N + threads - 1) // threads
        
        self.accel_kernel(
            self.pos_gpu, self.acc_gpu, self.nodes_gpu,
            np.int32(self.num_nodes_gpu), np.int32(self.N),
            np.float32(self.theta), np.float32(self.softening), np.float32(self.G),
            grid=(blocks, 1, 1), block=(threads, 1, 1)
        )
        self.update_pos_kernel(
            self.pos_gpu, self.vel_gpu, self.acc_gpu,
            np.float32(self.dt), np.int32(self.N),
            grid=(blocks, 1, 1), block=(threads, 1, 1)
        )
        self.new_acc_gpu, self.acc_gpu = self.acc_gpu, self.new_acc_gpu
        self.accel_kernel(
            self.pos_gpu, self.acc_gpu, self.nodes_gpu,
            np.int32(self.num_nodes_gpu), np.int32(self.N),
            np.float32(self.theta), np.float32(self.softening), np.float32(self.G),
            grid=(blocks, 1, 1), block=(threads, 1, 1)
        )
        self.update_vel_kernel(
            self.vel_gpu, self.new_acc_gpu, self.acc_gpu,
            np.float32(self.dt), np.int32(self.N),
            grid=(blocks, 1, 1), block=(threads, 1, 1)
        )
        drv.Context.synchronize()
    
    def get_positions(self):
        self.pos = self.pos_gpu.get().reshape(self.N, 3)
        return self.pos


def read_points_from_stdin():
    data = np.loadtxt(sys.stdin)
    if data.shape[1] != 6:
        raise ValueError(f"Expected 6 columns, got {data.shape[1]}")
    return data[:, :3], data[:, 3:6]


def main():
    positions, velocities = read_points_from_stdin()
    sim = NBodySimulationGPU()
    sim.set_data(positions, velocities)
    
    while True:
        try:
            sim.step()
            pos = sim.get_positions()
            sys.stdout.write(f"FRAME {sim.N}\n")
            for p in pos:
                sys.stdout.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            sys.stdout.flush()
        except BrokenPipeError:
            break


if __name__ == "__main__":
    main()