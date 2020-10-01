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

set(SOURCE_DIR "src/gnina_tensorflow_cpp")
include_directories(SOURCE_DIR)
set(SOURCES "${SOURCE_DIR}/calculate_distances.cpp")

add_subdirectory(lib/pybind11)

if (PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)
    pybind11_add_module(gnina_tensorflow_cpp
            "${SOURCE_DIR}/bindings.cpp"
            "${SOURCES}")
    target_include_directories(gnina_tensorflow_cpp PRIVATE include)
    target_link_libraries(gnina_tensorflow_cpp PRIVATE Eigen3::Eigen)
    set_target_properties(gnina_tensorflow_cpp PROPERTIES
            PREFIX ""
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gnina_tensorflow_cpp")
endif ()