#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "point.hpp"
#include "nanoflann_utils.hpp"

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
void get_nearest_neighbors(
    const np::ndarray &lidar_points,
    const np::ndarray &radar_points,
    const float max_pair_d2,
    np::ndarray &correspondences,
    np::ndarray &num_points)
{
    const pcl::PointCloud<vtr::lidar::PointWithInfo> raw_lidar = numpyToPointCloud(lidar_points);
    const pcl::PointCloud<vtr::lidar::PointWithInfo> query_points = numpyToPointCloud(radar_points);

    // create kd-tree of lidar data:
    vtr::lidar::NanoFLANNAdapter<vtr::lidar::PointWithInfo> adapter(raw_lidar);
    vtr::lidar::KDTreeParams tree_params(/* max leaf */ 10);
    auto kdtree = std::make_unique<vtr::lidar::KDTree<vtr::lidar::PointWithInfo>>(3, adapter, tree_params);
    kdtree->buildIndex();

    std::vector<std::pair<size_t, size_t>> sample_inds;
    sample_inds.resize(query_points.size());
    for (size_t i = 0; i < query_points.size(); i++)
        sample_inds[i].first = i;

    std::vector<float> nn_dists(sample_inds.size());

    vtr::lidar::KDTreeSearchParams search_params;
    for (size_t i = 0; i < sample_inds.size(); i++)
    {
        vtr::lidar::KDTreeResultSet result_set(1);
        result_set.init(&sample_inds[i].second, &nn_dists[i]);
        kdtree->findNeighbors(result_set, query_points[sample_inds[i].first].data, search_params);
    }
    std::vector<std::pair<size_t, size_t>> filtered_sample_inds;
    filtered_sample_inds.reserve(sample_inds.size());
    for (size_t i = 0; i < sample_inds.size(); i++)
    {
        if (nn_dists[i] < max_pair_d2)
        {
            filtered_sample_inds.push_back(sample_inds[i]);
        }
    }
    for (size_t i = 0; i < filtered_sample_inds.size(); i++)
    {
        correspondences[i][0] = filtered_sample_inds[i].first;
        correspondences[i][1] = filtered_sample_inds[i].second;
    }
    num_points[0] = filtered_sample_inds.size();
}

// boost python
BOOST_PYTHON_MODULE(get_nearest_neighbors)
{
    Py_Initialize();
    np::initialize();
    p::def("get_nearest_neighbors", get_nearest_neighbors);
}