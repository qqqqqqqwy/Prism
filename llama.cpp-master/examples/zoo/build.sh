export ANDROID_NDK_ROOT=/mnt/d/android-ndk-r26c

if [ ! -d "$ANDROID_NDK_ROOT" ]; then
    echo "Error: ANDROID_NDK_ROOT not found at $ANDROID_NDK_ROOT"
    exit 1
fi

BUILD_DIR="build"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "Building in: $(pwd)"

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-33 \
    -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-O3 -fno-finite-math-only -fopenmp -march=armv8.2-a+fp16+dotprod+i8mm -D__ARM_FEATURE_DOTPROD=1 -D__ARM_FEATURE_MATMUL_INT8=1" \
    -DCMAKE_CXX_FLAGS="-O3 -fno-finite-math-only -fopenmp -march=armv8.2-a+fp16+dotprod+i8mm -D__ARM_FEATURE_DOTPROD=1 -D__ARM_FEATURE_MATMUL_INT8=1"

make -j4

echo "Build complete. Executable is at: $(pwd)/llama-zoo"
