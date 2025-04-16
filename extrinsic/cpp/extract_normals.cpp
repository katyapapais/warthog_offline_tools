#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "normal.hpp"
#include "point.hpp"
#include "voxel_downsample.hpp"
#include <iostream>

namespace p = boost::python;
namespace np = boost::python::numpy;

pcl::PointCloud<vtr::lidar::PointWithInfo> numpyToPointCloud(const np::ndarray &np_in)
{
    uint row_size = np_in.shape(0);
    uint col_size = np_in.shape(1);
    pcl::PointCloud<vtr::lidar::PointWithInfo> cloud;
    cloud.reserve(row_size);
    for (size_t i = 0; i < row_size; i++)
    {
        vtr::lidar::PointWithInfo p;
        if (col_size >= 1)
            p.x = double(p::extract<float>(np_in[i][0]));
        if (col_size >= 2)
            p.y = double(p::extract<float>(np_in[i][1]));
        if (col_size >= 3)
            p.z = double(p::extract<float>(np_in[i][2]));
        p.rho = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        p.theta = std::atan2(std::sqrt(p.x * p.x + p.y * p.y), p.z);
        p.phi = std::atan2(p.y, p.x);
        cloud.push_back(p);
    }
    return cloud;
}

// Note: normals, and normal_scores have to be the appropriate size on the Python side of things...
void extract_normals(
    const np::ndarray &points_in,
    const float crop_range,
    const float voxel_size,
    const float polar_r_scale,
    const float vertical_angle_res,
    const float r_scale,
    const float h_scale,
    const int num_sample1,
    const float min_norm_score1,
    np::ndarray &points_out,
    np::ndarray &normals,
    np::ndarray &normal_scores,
    np::ndarray &num_points)
{
    const pcl::PointCloud<vtr::lidar::PointWithInfo> raw_point_cloud = numpyToPointCloud(points_in);
    // const auto point_cloud = raw_point_cloud.ptr();
    const auto point_cloud =
        std::make_shared<const pcl::PointCloud<vtr::lidar::PointWithInfo>>(raw_point_cloud);
    std::cout << "raw point cloud size: " << point_cloud->size();

    auto filtered_point_cloud =
        std::make_shared<pcl::PointCloud<vtr::lidar::PointWithInfo>>(*point_cloud);

    /// Range cropping
    {
        std::vector<int> indices;
        indices.reserve(filtered_point_cloud->size());
        for (size_t i = 0; i < filtered_point_cloud->size(); ++i)
        {
            if ((*filtered_point_cloud)[i].rho < crop_range)
                indices.emplace_back(i);
        }
        *filtered_point_cloud =
            pcl::PointCloud<vtr::lidar::PointWithInfo>(*filtered_point_cloud, indices);
    }

    std::cout << "range cropped point cloud size: " << filtered_point_cloud->size();

    vtr::lidar::voxelDownsample(*filtered_point_cloud, voxel_size);

    std::cout << "grid subsampled point cloud size: " << filtered_point_cloud->size();

    // Define the polar neighbors radius in the scaled polar coordinates
    float polar_r = polar_r_scale * vertical_angle_res;

    // Extracts normal vectors of sampled points
    auto norm_scores =
        vtr::lidar::extractNormal(point_cloud, filtered_point_cloud, polar_r,
                                  r_scale, h_scale, 1);

    /// Filtering based on normal scores (planarity + linearity)

    // Remove points with a low normal score
    auto sorted_norm_scores = norm_scores;
    std::sort(sorted_norm_scores.begin(), sorted_norm_scores.end());
    float min_score = sorted_norm_scores[std::max(
        0, (int)sorted_norm_scores.size() - num_sample1)];
    min_score = std::max(min_norm_score1, min_score);
    if (min_score >= 0)
    {
        std::vector<int> indices;
        indices.reserve(filtered_point_cloud->size());
        int i = 0;
        for (const auto &point : *filtered_point_cloud)
        {
            if (point.normal_score >= min_score)
                indices.emplace_back(i);
            i++;
        }
        *filtered_point_cloud =
            pcl::PointCloud<vtr::lidar::PointWithInfo>(*filtered_point_cloud, indices);
    }

    // normals to output...
    int i = 0;
    // for (size_t i = 0; i < filtered_point_cloud->size(); i++)
    for (const auto &point : *filtered_point_cloud)
    {
        points_out[i][0] = point.x;
        points_out[i][1] = point.y;
        points_out[i][2] = point.z;
        normals[i][0] = point.normal_x;
        normals[i][1] = point.normal_y;
        normals[i][2] = point.normal_z;
        normal_scores[i] = point.normal_score;
        i++;
    }
    num_points[0] = filtered_point_cloud->size();
}

// boost python
BOOST_PYTHON_MODULE(extract_normals)
{
    Py_Initialize();
    np::initialize();
    p::def("extract_normals", extract_normals);
}