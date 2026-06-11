cmake_minimum_required(VERSION 3.21)

set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm install path")
if(NOT ROCM_PATH)
  set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm install path" FORCE)
endif()
set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc" CACHE FILEPATH "HIP compiler" FORCE)

project(hk_iris_fused LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GPU_TARGET "CDNA4" CACHE STRING "GPU target architecture: CDNA3|CDNA4")
if(GPU_TARGET STREQUAL "CDNA3")
  set(GPU_FLAGS "-DKITTENS_CDNA3" "--offload-arch=gfx942")
elseif(GPU_TARGET STREQUAL "CDNA4")
  set(GPU_FLAGS "-DKITTENS_CDNA4" "--offload-arch=gfx950" "-DHIP_ENABLE_WARP_SYNC_BUILTINS" "-ffast-math")
else()
  message(FATAL_ERROR "Unknown GPU_TARGET: ${GPU_TARGET}")
endif()

set(HIPKITTENS_ROOT "$ENV{HIPKITTENS_SRC}" CACHE PATH "HipKittens checkout root")
if(NOT HIPKITTENS_ROOT)
  set(HIPKITTENS_ROOT "$ENV{HOME}/.cache/HipKittens" CACHE PATH "HipKittens checkout root" FORCE)
endif()
if(NOT EXISTS "${HIPKITTENS_ROOT}/include/kittens.cuh")
  message(FATAL_ERROR "HIPKITTENS_ROOT=${HIPKITTENS_ROOT} does not contain include/kittens.cuh")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

pybind11_add_module(hk_iris_fused kernel.cpp)
target_compile_options(hk_iris_fused PRIVATE
  ${GPU_FLAGS}
  -w
  -Rpass-analysis=kernel-resource-usage
)
target_include_directories(hk_iris_fused PRIVATE
  "${HIPKITTENS_ROOT}/include"
  "${HIPKITTENS_ROOT}/prototype"
  "${ROCM_PATH}/include/hip"
)
target_link_libraries(hk_iris_fused PRIVATE ${Python3_LIBRARIES} -ldl -lm)
set_target_properties(hk_iris_fused PROPERTIES
  PREFIX ""
  OUTPUT_NAME "hk_iris_fused"
  SUFFIX ".${Python3_SOABI}.so"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

