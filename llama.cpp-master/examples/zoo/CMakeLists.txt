cmake_minimum_required(VERSION 3.14)

project(llama_zoo)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -fno-finite-math-only")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fno-finite-math-only")

get_filename_component(LLAMA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
message(STATUS "LLAMA_ROOT: ${LLAMA_ROOT}")

add_definitions(-DGGML_VERSION="\\\"experiment\\\"")
add_definitions(-DGGML_COMMIT="\\\"unknown\\\"")
add_definitions(-DGGML_USE_CPU)

if (ANDROID)
    add_definitions(-DGGML_USE_OPENMP)
    add_definitions(-DGGML_NEON)
    add_definitions(-DGGML_ARM_FMA)
    add_definitions(-DGGML_ARM_DP_ASIMD)
    add_definitions(-D__ARM_NEON)
    add_definitions(-D__ARM_FEATURE_DOTPROD)
    add_definitions(-D__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    add_definitions(-D__ARM_FEATURE_MATMUL_INT8)
else()
    add_definitions(-D_XOPEN_SOURCE=600 -D_GNU_SOURCE)
endif()

include_directories(SYSTEM ${LLAMA_ROOT})
include_directories(SYSTEM ${LLAMA_ROOT}/include)
include_directories(SYSTEM ${LLAMA_ROOT}/src)
include_directories(SYSTEM ${LLAMA_ROOT}/ggml/include)
include_directories(SYSTEM ${LLAMA_ROOT}/ggml/src)
include_directories(SYSTEM ${LLAMA_ROOT}/ggml/src/ggml-cpu)

file(GLOB LLAMA_CORE_SRCS "${LLAMA_ROOT}/src/*.cpp")
set(GGUF_SRC "${LLAMA_ROOT}/ggml/src/gguf.cpp")
file(GLOB LLAMA_MODELS_SRCS "${LLAMA_ROOT}/src/models/*.cpp")

set(GGML_SRCS
    "${LLAMA_ROOT}/ggml/src/ggml.c"
    "${LLAMA_ROOT}/ggml/src/ggml-alloc.c"
    "${LLAMA_ROOT}/ggml/src/ggml-backend.cpp"
    "${LLAMA_ROOT}/ggml/src/ggml-backend-reg.cpp"
    "${LLAMA_ROOT}/ggml/src/ggml-quants.c"
    "${LLAMA_ROOT}/ggml/src/ggml-opt.cpp"
    "${LLAMA_ROOT}/ggml/src/ggml-threading.cpp"
)

file(GLOB GGML_CPU_ROOT_SRCS "${LLAMA_ROOT}/ggml/src/ggml-cpu/*.cpp" "${LLAMA_ROOT}/ggml/src/ggml-cpu/*.c")
file(GLOB GGML_CPU_ARM_SRCS
    "${LLAMA_ROOT}/ggml/src/ggml-cpu/arch/arm/*.cpp"
    "${LLAMA_ROOT}/ggml/src/ggml-cpu/arch/arm/*.c"
)

if (ANDROID)
    set(ARM_ARCH_FLAGS "-march=armv8.2-a+fp16+dotprod+i8mm")
    add_compile_options(${ARM_ARCH_FLAGS})

    set_source_files_properties(${GGML_CPU_ARM_SRCS} PROPERTIES COMPILE_FLAGS ${ARM_ARCH_FLAGS})
    set_source_files_properties(${GGML_CPU_ROOT_SRCS} PROPERTIES COMPILE_FLAGS ${ARM_ARCH_FLAGS})
    set_source_files_properties("${LLAMA_ROOT}/ggml/src/ggml-quants.c" PROPERTIES COMPILE_FLAGS ${ARM_ARCH_FLAGS})
endif()

set(ALL_SRCS
    ${LLAMA_CORE_SRCS}
    ${GGUF_SRC}
    ${LLAMA_MODELS_SRCS}
    ${GGML_SRCS}
    ${GGML_CPU_ROOT_SRCS}
    ${GGML_CPU_ARM_SRCS}
)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(llama-zoo
    llama-zoo.cpp
    ${ALL_SRCS}
)

if (ANDROID)
    target_link_libraries(llama-zoo log)
endif()

target_link_libraries(llama-zoo Threads::Threads dl)

if(OpenMP_CXX_FOUND)
    target_link_libraries(llama-zoo OpenMP::OpenMP_CXX)
endif()

