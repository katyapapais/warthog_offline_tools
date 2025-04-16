cmake_minimum_required(VERSION 3.16)
project(normalpy)

# Compile as C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-march=native -O3 -pthread -Wall -Wextra)


# find_package(ament_cmake REQUIRED)
# find_package(ament_cmake_python REQUIRED)
# find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(rclcpp REQUIRED)
# find_package(rclpy REQUIRED)
# find_package(pcl_conversions REQUIRED)
# find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED)

# execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
# find_package(OpenMP REQUIRED)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 ${OpenMP_CXX_FLAGS}")
# set(OMP_NUM_THREADS 8)

# set(DEPEND_LIBS "boost_system")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-unused-function")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# packages
find_package(PythonInterp)
# find_package(Boost REQUIRED COMPONENTS system thread chrono timer)
find_package(Boost REQUIRED COMPONENTS
    python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}
    numpy${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}
    system
    thread
    chrono
    timer
)
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)




# openmp
#find_package(OpenMP REQUIRED)
#set(OMP_NUM_THREADS 8)

# steam
# find_package(Steam REQUIRED)

# opencv
# find_package(OpenCV 3 REQUIRED)

# boost python
ENABLE_TESTING()
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

# single function call version
PYTHON_ADD_MODULE(extract_normals extract_normals.cpp)
target_link_libraries(extract_normals PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

PYTHON_ADD_MODULE(get_nearest_neighbors get_nearest_neighbors.cpp)
target_link_libraries(get_nearest_neighbors PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

# class version
# PYTHON_ADD_MODULE(SteamSolver SteamSolver.cpp P2P3ErrorEval.cpp SE2VelPriorEval.cpp mcransac.cpp)
# target_link_libraries(SteamSolver PRIVATE ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${Steam_LIBRARIES})

# dataloader
# PYTHON_ADD_MODULE(DataLoader DataLoader.cpp)
# target_link_libraries(DataLoader PRIVATE ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${Steam_LIBRARIES} ${OpenCV_LIBRARIES})