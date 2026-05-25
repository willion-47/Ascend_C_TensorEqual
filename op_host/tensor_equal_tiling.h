#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TensorEqualTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, totalSize);
  TILING_DATA_FIELD_DEF(uint32_t, dimNum);
  TILING_DATA_FIELD_DEF(uint32_t, dataType);
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF(uint8_t, isBroadcast); // 0: 无广播(快路径), 1: 有广播(安全路径)
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);  // Host端动态算出的单次最优UB切片长度
  TILING_DATA_FIELD_DEF_ARR(uint64_t, 25, yShape);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, 25, yStride);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, 25, x1Stride);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, 25, x2Stride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TensorEqual, TensorEqualTilingData)
}