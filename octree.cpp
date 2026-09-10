#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstring>

namespace py = pybind11;

struct Node {
    float cx, cy, cz;
    float mass;
    float size;
    float min_x, max_x;
    float min_y, max_y;
    float min_z, max_z;
    int children[8];
    int count;
    int particle;
    int is_leaf;
    int parent;
};

int getOctant(float px, float py, float pz, const Node& node) {
    float mid_x = (node.min_x + node.max_x) * 0.5f;
    float mid_y = (node.min_y + node.max_y) * 0.5f;
    float mid_z = (node.min_z + node.max_z) * 0.5f;
    
    int octant = 0;
    if (px >= mid_x) octant |= 1;
    if (py >= mid_y) octant |= 2;
    if (pz >= mid_z) octant |= 4;
    return octant;
}

void insertParticle(int idx, float px, float py, float pz, float m, std::vector<Node>& nodes) {
    int node_idx = 0;
    int iter = 0;
    
    while (iter++ < 1000) {
        Node* node = &nodes[node_idx];
        
        if (node->is_leaf && node->particle != -1) {
            int old_p = node->particle;
            float old_x = node->cx;
            float old_y = node->cy;
            float old_z = node->cz;
            float old_m = node->mass;
            
            float mid_x = (node->min_x + node->max_x) * 0.5f;
            float mid_y = (node->min_y + node->max_y) * 0.5f;
            float mid_z = (node->min_z + node->max_z) * 0.5f;
            
            int base_idx = nodes.size();
            nodes.resize(nodes.size() + 8);
            
            node = &nodes[node_idx];
            
            node->mass = 0.0f;
            node->count = 8;
            node->is_leaf = 0;
            node->particle = -1;
            
            for (int i = 0; i < 8; i++) {
                int child_idx = base_idx + i;
                node->children[i] = child_idx;
                
                float min_x2 = (i & 1) ? mid_x : node->min_x;
                float max_x2 = (i & 1) ? node->max_x : mid_x;
                float min_y2 = (i & 2) ? mid_y : node->min_y;
                float max_y2 = (i & 2) ? node->max_y : mid_y;
                float min_z2 = (i & 4) ? mid_z : node->min_z;
                float max_z2 = (i & 4) ? node->max_z : mid_z;
                
                nodes[child_idx].min_x = min_x2;
                nodes[child_idx].max_x = max_x2;
                nodes[child_idx].min_y = min_y2;
                nodes[child_idx].max_y = max_y2;
                nodes[child_idx].min_z = min_z2;
                nodes[child_idx].max_z = max_z2;
                nodes[child_idx].cx = (min_x2 + max_x2) * 0.5f;
                nodes[child_idx].cy = (min_y2 + max_y2) * 0.5f;
                nodes[child_idx].cz = (min_z2 + max_z2) * 0.5f;
                nodes[child_idx].size = max_x2 - min_x2;
                nodes[child_idx].mass = 0.0f;
                nodes[child_idx].count = 0;
                nodes[child_idx].particle = -1;
                nodes[child_idx].is_leaf = 1;
                nodes[child_idx].parent = node_idx;
                for (int j = 0; j < 8; j++) {
                    nodes[child_idx].children[j] = -1;
                }
            }
            
            int old_octant = getOctant(old_x, old_y, old_z, *node);
            int old_child = node->children[old_octant];
            nodes[old_child].mass = old_m;
            nodes[old_child].cx = old_x;
            nodes[old_child].cy = old_y;
            nodes[old_child].cz = old_z;
            nodes[old_child].particle = old_p;
            nodes[old_child].is_leaf = 1;
            
            node->mass += old_m;
            
            int new_octant = getOctant(px, py, pz, *node);
            node_idx = node->children[new_octant];
            continue;
        }
        
        if (node->is_leaf && node->particle == -1) {
            node->mass = m;
            node->cx = px;
            node->cy = py;
            node->cz = pz;
            node->particle = idx;
            
            int current = node->parent;
            while (current != -1) {
                nodes[current].mass += m;
                current = nodes[current].parent;
            }
            break;
        }
        
        if (!node->is_leaf) {
            int octant = getOctant(px, py, pz, *node);
            node_idx = node->children[octant];
            continue;
        }
        
        break;
    }
}

void computeCenterOfMass(int node_idx, std::vector<Node>& nodes) {
    Node* node = &nodes[node_idx];
    
    if (node->is_leaf) {
        return;
    }
    
    float total_mass = 0.0f;
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    
    for (int i = 0; i < 8; i++) {
        int child_idx = node->children[i];
        if (child_idx >= 0 && child_idx < (int)nodes.size()) {
            computeCenterOfMass(child_idx, nodes);
            Node* child = &nodes[child_idx];
            if (child->mass > 0.0f) {
                total_mass += child->mass;
                cx += child->mass * child->cx;
                cy += child->mass * child->cy;
                cz += child->mass * child->cz;
            }
        }
    }
    
    if (total_mass > 0.0f) {
        node->mass = total_mass;
        node->cx = cx / total_mass;
        node->cy = cy / total_mass;
        node->cz = cz / total_mass;
    }
}

std::vector<Node> build_tree_internal(float* ptr_x, float* ptr_y, float* ptr_z, 
                                       float* ptr_mass, int N, float& root_size) {
    float min_x = ptr_x[0], max_x = ptr_x[0];
    float min_y = ptr_y[0], max_y = ptr_y[0];
    float min_z = ptr_z[0], max_z = ptr_z[0];
    
    for (int i = 1; i < N; i++) {
        if (ptr_x[i] < min_x) min_x = ptr_x[i];
        if (ptr_x[i] > max_x) max_x = ptr_x[i];
        if (ptr_y[i] < min_y) min_y = ptr_y[i];
        if (ptr_y[i] > max_y) max_y = ptr_y[i];
        if (ptr_z[i] < min_z) min_z = ptr_z[i];
        if (ptr_z[i] > max_z) max_z = ptr_z[i];
    }
    
    float pad = 0.1f;
    min_x -= pad; max_x += pad;
    min_y -= pad; max_y += pad;
    min_z -= pad; max_z += pad;
    
    root_size = max_x - min_x;
    
    std::vector<Node> nodes;
    
    Node root;
    root.min_x = min_x; root.max_x = max_x;
    root.min_y = min_y; root.max_y = max_y;
    root.min_z = min_z; root.max_z = max_z;
    root.cx = (min_x + max_x) * 0.5f;
    root.cy = (min_y + max_y) * 0.5f;
    root.cz = (min_z + max_z) * 0.5f;
    root.size = root_size;
    root.mass = 0.0f;
    root.count = 0;
    root.particle = -1;
    root.is_leaf = 1;
    root.parent = -1;
    for (int i = 0; i < 8; i++) root.children[i] = -1;
    nodes.push_back(root);
    
    for (int i = 0; i < N; i++) {
        insertParticle(i, ptr_x[i], ptr_y[i], ptr_z[i], ptr_mass[i], nodes);
    }
    
    computeCenterOfMass(0, nodes);
    
    return nodes;
}

py::array_t<float> build_octree_gpu(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<float, py::array::c_style | py::array::forcecast> y,
    py::array_t<float, py::array::c_style | py::array::forcecast> z,
    py::array_t<float, py::array::c_style | py::array::forcecast> mass) {
    
    py::buffer_info buf_x = x.request();
    py::buffer_info buf_y = y.request();
    py::buffer_info buf_z = z.request();
    py::buffer_info buf_mass = mass.request();
    
    int N = buf_x.shape[0];
    float* ptr_x = static_cast<float*>(buf_x.ptr);
    float* ptr_y = static_cast<float*>(buf_y.ptr);
    float* ptr_z = static_cast<float*>(buf_z.ptr);
    float* ptr_mass = static_cast<float*>(buf_mass.ptr);
    
    float root_size;
    std::vector<Node> nodes = build_tree_internal(ptr_x, ptr_y, ptr_z, ptr_mass, N, root_size);
    
    int num_nodes = nodes.size();
    
    py::array_t<float> result({num_nodes, 11});
    auto buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    
    for (int i = 0; i < num_nodes; i++) {
        Node& node = nodes[i];
        int offset = i * 11;
        
        ptr[offset + 0] = node.cx;
        ptr[offset + 1] = node.cy;
        ptr[offset + 2] = node.cz;
        ptr[offset + 3] = node.size;
        ptr[offset + 4] = node.mass;
        ptr[offset + 5] = (node.min_x + node.max_x) * 0.5f;
        ptr[offset + 6] = (node.min_y + node.max_y) * 0.5f;
        ptr[offset + 7] = (node.min_z + node.max_z) * 0.5f;
        
        int first_child = -1;
        for (int j = 0; j < 8; j++) {
            if (node.children[j] >= 0) {
                first_child = node.children[j];
                break;
            }
        }
        ptr[offset + 8] = (float)first_child;
        
        ptr[offset + 9] = (float)node.is_leaf;
        ptr[offset + 10] = (node.mass == 0.0f && !node.is_leaf) ? 1.0f : 0.0f;
    }
    
    return result;
}

py::dict build_octree_info(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<float, py::array::c_style | py::array::forcecast> y,
    py::array_t<float, py::array::c_style | py::array::forcecast> z,
    py::array_t<float, py::array::c_style | py::array::forcecast> mass) {
    
    py::buffer_info buf_x = x.request();
    py::buffer_info buf_y = y.request();
    py::buffer_info buf_z = z.request();
    py::buffer_info buf_mass = mass.request();
    
    int N = buf_x.shape[0];
    float* ptr_x = static_cast<float*>(buf_x.ptr);
    float* ptr_y = static_cast<float*>(buf_y.ptr);
    float* ptr_z = static_cast<float*>(buf_z.ptr);
    float* ptr_mass = static_cast<float*>(buf_mass.ptr);
    
    float root_size;
    std::vector<Node> nodes = build_tree_internal(ptr_x, ptr_y, ptr_z, ptr_mass, N, root_size);
    
    py::dict info;
    info["num_nodes"] = (int)nodes.size();
    info["root_size"] = root_size;
    info["root_mass"] = nodes[0].mass;
    info["root_cx"] = nodes[0].cx;
    info["root_cy"] = nodes[0].cy;
    info["root_cz"] = nodes[0].cz;
    
    return info;
}

PYBIND11_MODULE(octree_lib, m) {
    m.doc() = "Octree library for fast spatial partitioning";
    
    m.def("build_octree_gpu", &build_octree_gpu,
          "Build octree, returns GPU-ready numpy array [num_nodes, 11]\n"
          "Columns: com_x, com_y, com_z, size, mass, center_x, center_y, center_z, child_index, is_leaf, is_empty",
          py::arg("x"), py::arg("y"), py::arg("z"), py::arg("mass"));
    
    m.def("build_octree_info", &build_octree_info,
          "Build octree, returns basic info dict",
          py::arg("x"), py::arg("y"), py::arg("z"), py::arg("mass"));
}