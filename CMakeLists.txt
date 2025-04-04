cmake_minimum_required(VERSION 3.26)

# When building directly using CMake, make sure you run the install step
# (it places the .so files in the correct location).
#
# Example:
# mkdir build && cd build
# cmake -G Ninja -DVLLM_PYTHON_EXECUTABLE=`which python3` -DCMAKE_INSTALL_PREFIX=.. ..
# cmake --build . --target install
#
# If you want to only build one target, make sure to install it manually:
# cmake --build . --target _C
# cmake --install . --component _C
project(vllm_extensions LANGUAGES CXX)

# CUDA by default, can be overridden by using -DVLLM_TARGET_DEVICE=... (used by setup.py)
set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

# Prevent installation of dependencies (cutlass) by default.
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11" "3.12")

# Supported NVIDIA architectures.
set(CUDA_SUPPORTED_ARCHS "9.0")

# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100")

#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: the CUDA torch version is derived from pyproject.toml and various
# requirements.txt files and should be kept consistent.  The ROCm torch
# versions are derived from Dockerfile.rocm
#
set(TORCH_SUPPORTED_VERSION_CUDA "2.4.0")
set(TORCH_SUPPORTED_VERSION_ROCM "2.5.0")

#
# Try to find python package with an executable that exactly matches
# `VLLM_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

# Ensure the 'nvcc' command is in the PATH
find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)

#
message(STATUS "Enabling core extension.")

# Define _core_C extension
#  built for (almost) every target platform, (excludes TPU and Neuron)

set(VLLM_EXT_SRC
  "csrc/core/torch_bindings.cpp")

define_gpu_extension_target(
  _core_C
  DESTINATION vllm
  LANGUAGE CXX
  SOURCES ${VLLM_EXT_SRC}
  COMPILE_FLAGS ${CXX_COMPILE_FLAGS}
  USE_SABI 3
  WITH_SOABI)

#
# Forward the non-CUDA device extensions to external CMake scripts.
#
if (NOT VLLM_TARGET_DEVICE STREQUAL "cuda" AND
    NOT VLLM_TARGET_DEVICE STREQUAL "rocm")
    if (VLLM_TARGET_DEVICE STREQUAL "cpu")
        include(${CMAKE_CURRENT_LIST_DIR}/cmake/cpu_extension.cmake)
    else()
        return()
    endif()
    return()
endif()

#
# Set up GPU language and check the torch version and warn if it isn't
# what is expected.
#
if (NOT HIP_FOUND AND CUDA_FOUND)
  set(VLLM_GPU_LANG "CUDA")

  if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
      "expected for CUDA build, saw ${Torch_VERSION} instead.")
  endif()
elseif(HIP_FOUND)
  set(VLLM_GPU_LANG "HIP")

  # Importing torch recognizes and sets up some HIP/ROCm configuration but does
  # not let cmake recognize .hip files. In order to get cmake to understand the
  # .hip extension automatically, HIP must be enabled explicitly.
  enable_language(HIP)

  # ROCm 5.X and 6.X
  if (ROCM_VERSION_DEV_MAJOR GREATER_EQUAL 5 AND
      NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_ROCM})
    message(WARNING "Pytorch version >= ${TORCH_SUPPORTED_VERSION_ROCM} "
      "expected for ROCm build, saw ${Torch_VERSION} instead.")
  endif()
else()
  message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif()

#
# Override the GPU architectures detected by cmake/torch and filter them by
# the supported versions for the current language.
# The final set of arches is stored in `VLLM_GPU_ARCHES`.
#
override_gpu_arches(VLLM_GPU_ARCHES
  ${VLLM_GPU_LANG}
  "${${VLLM_GPU_LANG}_SUPPORTED_ARCHS}")

#
# Query torch for additional GPU compilation flags for the given
# `VLLM_GPU_LANG`.
# The final set of arches is stored in `VLLM_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})

#
# Set nvcc parallelism.
#
if(NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()


#
# Use FetchContent for C++ dependencies that are compiled as part of vLLM's build process.
# Configure it to place files in vllm/.deps, in order to play nicely with sccache.
#
include(FetchContent)
get_filename_component(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
file(MAKE_DIRECTORY "${FETCHCONTENT_BASE_DIR}")
set(FETCHCONTENT_BASE_DIR "${PROJECT_ROOT_DIR}/.deps")
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

#
# Define other extension targets
#

#
# _C extension
#

set(VLLM_EXT_SRC
  "csrc/cache_kernels.cu"
  "csrc/attention/attention_kernels.cu"
  "csrc/pos_encoding_kernels.cu"
  "csrc/activation_kernels.cu"
  "csrc/layernorm_kernels.cu"
  "csrc/quantization/gptq/q_gemm.cu"
  "csrc/quantization/compressed_tensors/int8_quant_kernels.cu"
  "csrc/quantization/fp8/common.cu"
  "csrc/cuda_utils_kernels.cu"
  "csrc/moe_align_block_size_kernels.cu"
  "csrc/prepare_inputs/advance_step.cu"
  "csrc/torch_bindings.cpp")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  SET(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

  # Set CUTLASS_REVISION manually -- its revision detection doesn't work in this case.
  set(CUTLASS_REVISION "v3.5.1" CACHE STRING "CUTLASS revision to use")

  FetchContent_Declare(
        cutlass
        GIT_REPOSITORY /workspace/xbzhuang/cutlass
        GIT_TAG v3.5.1
        GIT_PROGRESS TRUE

        # Speed up CUTLASS download by retrieving only the specified GIT_TAG instead of the history.
        # Important: If GIT_SHALLOW is enabled then GIT_TAG works only with branch names and tags.
        # So if the GIT_TAG above is updated to a commit hash, GIT_SHALLOW must be set to FALSE
        GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(cutlass)

  list(APPEND VLLM_EXT_SRC
    "csrc/mamba/mamba_ssm/selective_scan_fwd.cu"
    "csrc/mamba/causal_conv1d/causal_conv1d.cu"
    "csrc/quantization/aqlm/gemm_kernels.cu"
    "csrc/quantization/awq/gemm_kernels.cu"
    "csrc/quantization/marlin/dense/marlin_cuda_kernel.cu"
    "csrc/quantization/marlin/sparse/marlin_24_cuda_kernel.cu"
    "csrc/quantization/marlin/qqq/marlin_qqq_gemm_kernel.cu"
    "csrc/quantization/gptq_marlin/gptq_marlin.cu"
    "csrc/quantization/gptq_marlin/gptq_marlin_repack.cu"
    "csrc/quantization/gptq_marlin/awq_marlin_repack.cu"
    "csrc/quantization/gguf/gguf_kernel.cu"
    "csrc/quantization/fp8/fp8_marlin.cu"
    "csrc/custom_all_reduce.cu"
    "csrc/permute_cols.cu"
    "csrc/quantization/cutlass_w8a8/scaled_mm_entry.cu"
    "csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cu"
    "csrc/quantization/cutlass_w8a8/scaled_mm_c3x.cu")

  #
  # The CUTLASS kernels for Hopper require sm90a to be enabled.
  # This is done via the below gencode option, BUT that creates kernels for both sm90 and sm90a.
  # That adds an extra 17MB to compiled binary, so instead we selectively enable it.
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
    set_source_files_properties(
          "csrc/quantization/cutlass_w8a8/scaled_mm_c3x.cu"
          PROPERTIES
          COMPILE_FLAGS
          "-gencode arch=compute_90a,code=sm_90a")
  endif()


  #
  # Machete kernels

  # The machete kernels only work on hopper and require CUDA 12.0 or later.
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
    #
    # For the Machete kernels we automatically generate sources for various 
    # preselected input type pairs and schedules.
    # Generate sources:
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env 
      PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass_extensions/:${CUTLASS_DIR}/python/:${VLLM_PYTHON_PATH}:$PYTHONPATH 
        ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantization/machete/generate.py
      RESULT_VARIABLE machete_generation_result
      OUTPUT_VARIABLE machete_generation_output
      OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
      ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
    )

    if (NOT machete_generation_result EQUAL 0)
      message(FATAL_ERROR "Machete generation failed."
                          " Result: \"${machete_generation_result}\"" 
                          "\nCheck the log for details: "
                          "${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log")
    else()
      message(STATUS "Machete generation completed successfully.")
    endif()

    # Add machete generated sources
    file(GLOB MACHETE_GEN_SOURCES "csrc/quantization/machete/generated/*.cu")
    list(APPEND VLLM_EXT_SRC ${MACHETE_GEN_SOURCES})
    message(STATUS "Machete generated sources: ${MACHETE_GEN_SOURCES}")

    set_source_files_properties(
          ${MACHETE_GEN_SOURCES}
          PROPERTIES
          COMPILE_FLAGS
          "-gencode arch=compute_90a,code=sm_90a")
  endif()

  # Add pytorch binding for machete (add on even CUDA < 12.0 so that we can
  #  raise an error if the user that this was built with an incompatible 
  #  CUDA version)
  list(APPEND VLLM_EXT_SRC
    csrc/quantization/machete/machete_pytorch.cu)
endif()

message(STATUS "Enabling C extension.")
define_gpu_extension_target(
  _C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

# If CUTLASS is compiled on NVCC >= 12.5, it by default uses 
# cudaGetDriverEntryPointByVersion as a wrapper to avoid directly calling the 
# driver API. This causes problems when linking with earlier versions of CUDA.
# Setting this variable sidesteps the issue by calling the driver directly.
target_compile_definitions(_C PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)

#
# _moe_C extension
#

set(VLLM_MOE_EXT_SRC
  "csrc/moe/torch_bindings.cpp"
  "csrc/moe/topk_softmax_kernels.cu")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_MOE_EXT_SRC
      "csrc/moe/marlin_kernels/marlin_moe_kernel.h"
      "csrc/moe/marlin_kernels/marlin_moe_kernel_ku4b8.h"
      "csrc/moe/marlin_kernels/marlin_moe_kernel_ku4b8.cu"
      "csrc/moe/marlin_kernels/marlin_moe_kernel_ku8b128.h"
      "csrc/moe/marlin_kernels/marlin_moe_kernel_ku8b128.cu"
      "csrc/moe/marlin_moe_ops.cu")
endif()

message(STATUS "Enabling moe extension.")
define_gpu_extension_target(
  _moe_C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_MOE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)

if(VLLM_GPU_LANG STREQUAL "HIP")
  #
  # _rocm_C extension
  #
  set(VLLM_ROCM_EXT_SRC
    "csrc/rocm/torch_bindings.cpp"
    "csrc/rocm/attention.cu")

  define_gpu_extension_target(
    _rocm_C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_ROCM_EXT_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    USE_SABI 3
    WITH_SOABI)
endif()

# vllm-flash-attn currently only supported on CUDA
if (NOT VLLM_TARGET_DEVICE STREQUAL "cuda")
  return()
endif ()

#
# Build vLLM flash attention from source
#
# IMPORTANT: This has to be the last thing we do, because vllm-flash-attn uses the same macros/functions as vLLM.
# Because functions all belong to the global scope, vllm-flash-attn's functions overwrite vLLMs.
# They should be identical but if they aren't, this is a massive footgun.
#
# The vllm-flash-attn install rules are nested under vllm to make sure the library gets installed in the correct place.
# To only install vllm-flash-attn, use --component vllm_flash_attn_c.
# If no component is specified, vllm-flash-attn is still installed.

# If VLLM_FLASH_ATTN_SRC_DIR is set, vllm-flash-attn is installed from that directory instead of downloading.
# This is to enable local development of vllm-flash-attn within vLLM.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{VLLM_FLASH_ATTN_SRC_DIR})
  set(VLLM_FLASH_ATTN_SRC_DIR $ENV{VLLM_FLASH_ATTN_SRC_DIR})
endif()

if(VLLM_FLASH_ATTN_SRC_DIR)
  FetchContent_Declare(vllm-flash-attn SOURCE_DIR ${VLLM_FLASH_ATTN_SRC_DIR})
else()
  FetchContent_Declare(
          vllm-flash-attn
          GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git
          GIT_TAG 013f0c4fc47e6574060879d9734c1df8c5c273bd
          GIT_PROGRESS TRUE
  )
endif()

# Set the parent build flag so that the vllm-flash-attn library does not redo compile flag and arch initialization.
set(VLLM_PARENT_BUILD ON)

# Ensure the vllm/vllm_flash_attn directory exists before installation
install(CODE "file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn\")" COMPONENT vllm_flash_attn_c)

# Make sure vllm-flash-attn install rules are nested under vllm/
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY FALSE)" COMPONENT vllm_flash_attn_c)
install(CODE "set(OLD_CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}\")" COMPONENT vllm_flash_attn_c)
install(CODE "set(CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}/vllm/\")" COMPONENT vllm_flash_attn_c)

# Fetch the vllm-flash-attn library
FetchContent_MakeAvailable(vllm-flash-attn)
message(STATUS "vllm-flash-attn is available at ${vllm-flash-attn_SOURCE_DIR}")

# Restore the install prefix
install(CODE "set(CMAKE_INSTALL_PREFIX \"\${OLD_CMAKE_INSTALL_PREFIX}\")" COMPONENT vllm_flash_attn_c)
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" COMPONENT vllm_flash_attn_c)

# Copy over the vllm-flash-attn python files
install(
        DIRECTORY ${vllm-flash-attn_SOURCE_DIR}/vllm_flash_attn/
        DESTINATION vllm/vllm_flash_attn
        COMPONENT vllm_flash_attn_c
        FILES_MATCHING PATTERN "*.py"
)

# Nothing after vllm-flash-attn, see comment about macros above
