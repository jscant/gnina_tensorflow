#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

// Calculate the Euclidean distance (L2 norm) between two vectors.
template <typename T>
float get_distance(const std::vector<T> &v1, const std::vector<T> &v2);

template<typename T>
float get_distance(const std::vector<T> &v1, const std::vector<T> &v2) {
    float total_dist = 0;
    for (unsigned int i = 0; i < v1.size(); ++i) {
        total_dist += pow(v1.at(i) - v2.at(i), 2);
    }
    return sqrt(total_dist);
}

// Return a 3D Eigen::Tensor of floats, with each entry denoting the minimum L2 distance from that point to any part of
// the input where there is a non-zero input in a ligand channel.
Eigen::Tensor<float, 3> calculate_ligand_distances(
        int rec_channels,
        Eigen::Tensor<float, 4> input_tensor,
        float point_dis);