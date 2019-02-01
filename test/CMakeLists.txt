cmake_minimum_required(VERSION 3.8)
project(test_PauliExpression LANGUAGES CXX)
enable_testing()

# set helper pathes to find libraries and packages
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")

# Project paths
# =============

set(project_include_dir "${CMAKE_CURRENT_LIST_DIR}/../include")
set(project_source_dir "${CMAKE_CURRENT_LIST_DIR}/../source")
set(project_bin_dir "${CMAKE_CURRENT_LIST_DIR}/../bin")

# Language Flags
# ==============

# enforce C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)

# Find Catch2
# ===========

find_package(Catch2 REQUIRED)

# Find Python
# ===========

if(DEFINED ENV{PYTHONHOME})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{PYTHONHOME}")
endif()
find_package(PythonLibs REQUIRED)
set(CXX_INCLUDES ${CXX_INCLUDES} ${PYTHON_INCLUDE_DIR})

# Find NumPy
# ==========

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake" BEFORE)

find_package(NumPy REQUIRED)
set(CXX_INCLUDES ${CXX_INCLUDES} ${NUMPY_INCLUDE_DIRS})

# Compiler flags
# ==============

# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Compile PauliExpression
# =======================

add_executable(tests test.cpp)
target_include_directories(tests PRIVATE ${project_include_dir})
target_include_directories(tests PRIVATE ${CATCH2_INCLUDE_DIRS})
target_include_directories(tests PRIVATE ${CXX_INCLUDES})
target_link_libraries(tests Catch2::Catch2)

add_test(NAME tests COMMAND test_executable)

include(CTest)
include(Catch)
catch_discover_tests(tests)

