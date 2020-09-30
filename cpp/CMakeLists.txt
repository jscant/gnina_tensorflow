if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
project(gnina_tensorflow_cpp)
cmake_minimum_required(VERSION 3.0)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif ()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(SOURCE_DIR "src/calculate_distances")
include_directories(SOURCE_DIR)
set(SOURCES "${SOURCE_DIR}/calculate_distances.cpp")

add_subdirectory(lib/pybind11)

if (PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)
    pybind11_add_module(calculate_distances
            "${SOURCE_DIR}/bindings.cpp"
            "${SOURCES}")
    target_include_directories(calculate_distances PRIVATE include)
    target_link_libraries(calculate_distances PRIVATE Eigen3::Eigen)
    set_target_properties(calculate_distances PROPERTIES
            PREFIX ""
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/calculate_distances")
endif ()