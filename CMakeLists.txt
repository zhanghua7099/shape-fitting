# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

# Important!!! 
# /usr/bin/ld: libfitting_library.a(cylinder_fitting_hough.cpp.o): relocation R_X86_64_PC32 against symbol `_ZN5boost6detail15sp_counted_base7destroyEv' can not be used when making a shared object; recompile with -fPIC
# /usr/bin/ld: final link failed: bad value
# collect2: error: ld returned 1 exit status
# ninja: build stopped: subcommand failed.
add_compile_options(-fPIC)

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package (Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost COMPONENTS system filesystem regex REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS})

add_library (fitting_library src/cylinder_fitting.cpp src/cylinder_fitting_hough.cpp src/gaussian_sphere.cpp src/fitting_data.cpp)

# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(_core MODULE src/main.cpp WITH_SOABI)
target_link_libraries(_core PRIVATE pybind11::headers fitting_library ${Eigen_LIBS} ${PCL_LIBRARIES})

# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION cy_fit_hough)
