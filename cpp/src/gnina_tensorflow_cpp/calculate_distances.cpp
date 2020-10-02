#include <limits>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "calculate_distances.h"

Eigen::Tensor<float, 5> calculate_ligand_distances(
        int rec_channels, Eigen::Tensor<float, 5> input_tensor, float point_dis
) {

    const Eigen::Tensor<int, 5>::Dimensions &dims = input_tensor.dimensions();
    const int batches = dims.at(0);
    const int channels = dims.at(1);
    const int x = dims.at(2);
    const int y = dims.at(3);
    const int z = dims.at(4);
    const int lig_channels = channels - rec_channels;

    Eigen::array<int, 5> offsets = {0, rec_channels, 0, 0, 0};
    Eigen::array<int, 5> extents = {batches, lig_channels, x, y, z};
    Eigen::Tensor<float, 5> ligand_tensor = input_tensor.slice(offsets,
                                                               extents);

    Eigen::Tensor<float, 5> result(batches, channels, x, y, z);

    const Eigen::Tensor<int, 1>::Dimensions &channel_dim =
            Eigen::Tensor<int, 1>::Dimensions(1);
    Eigen::Tensor<float, 4> reduced_ligand_tensor = ligand_tensor.sum(
            channel_dim);

    for (auto batch = 0; batch < batches; ++batch) {
        for (auto i = 0; i < x; ++i) {
            for (auto j = 0; j < y; ++j) {
                for (auto k = 0; k < z; ++k) {
                    const std::vector<int> coords({i, j, k});
                    int cube_size = 3;
                    float min_dist = std::numeric_limits<float>::max();
                    while (cube_size <= 2 * x + 1) {
                        const int radius = cube_size / 2 + 1;
                        const int imin = std::max(0, i - radius);
                        const int imax = std::min(x, i + radius);
                        const int jmin = std::max(0, j - radius);
                        const int jmax = std::min(y, j + radius);
                        const int kmin = std::max(0, k - radius);
                        const int kmax = std::min(z, k + radius);

                        for (auto cube_i = imin; cube_i < imax; ++cube_i) {
                            for (auto cube_j = jmin; cube_j < jmax; ++cube_j) {
                                for (auto cube_k = kmin;
                                     cube_k < kmax; ++cube_k) {
                                    if (reduced_ligand_tensor(
                                            batch,
                                            cube_i,
                                            cube_j,
                                            cube_k) > 0) {
                                        const std::vector<int>
                                                candidate_coords = {
                                                cube_i, cube_j, cube_k
                                        };
                                        min_dist = std::min(
                                                min_dist,
                                                get_distance(candidate_coords,
                                                             coords));
                                    }
                                }
                            }
                        }

                        if (min_dist < std::numeric_limits<float>::max()) {
                            for (auto channel = 0;
                                 channel < channels; ++channel) {
                                result(batch, channel, i, j, k) = min_dist;
                            }
                            break;
                        }
                        cube_size = cube_size + 2;
                    }
                }
            }
        }
    }

    return
            result * point_dis;
}
