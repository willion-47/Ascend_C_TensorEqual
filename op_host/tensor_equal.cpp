#include "tensor_equal_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr size_t kMaxDims = 25;
constexpr uint32_t kDefaultBlockDim = 8;

bool BuildBroadcastShape(const gert::Shape& x1Shape, const gert::Shape& x2Shape, gert::Shape& yShape)
{
    const size_t x1DimNum = x1Shape.GetDimNum();
    const size_t x2DimNum = x2Shape.GetDimNum();
    const size_t dimNum = x1DimNum > x2DimNum ? x1DimNum : x2DimNum;
    if (dimNum > kMaxDims) {
        return false;
    }

    yShape.SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        const size_t x1Offset = dimNum - x1DimNum;
        const size_t x2Offset = dimNum - x2DimNum;
        const int64_t x1Dim = i < x1Offset ? 1 : x1Shape.GetDim(i - x1Offset);
        const int64_t x2Dim = i < x2Offset ? 1 : x2Shape.GetDim(i - x2Offset);

        if (x1Dim <= 0 || x2Dim <= 0) {
            return false;
        }
        if (x1Dim != x2Dim && x1Dim != 1 && x2Dim != 1) {
            return false;
        }
        yShape[i] = x1Dim > x2Dim ? x1Dim : x2Dim;
    }
    return true;
}

uint64_t ShapeSize(const gert::Shape& shape)
{
    uint64_t size = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        size *= static_cast<uint64_t>(shape.GetDim(i));
    }
    return size;
}

void BuildStrides(const gert::Shape& inputShape, const gert::Shape& yShape, uint64_t* inputStride, uint64_t* yStride)
{
    const size_t dimNum = yShape.GetDimNum();
    const size_t inputDimNum = inputShape.GetDimNum();
    const size_t inputOffset = dimNum - inputDimNum;

    uint64_t baseStride[kMaxDims] = {0};
    uint64_t stride = 1;
    for (size_t rev = 0; rev < inputDimNum; ++rev) {
        const size_t idx = inputDimNum - 1 - rev;
        baseStride[idx] = stride;
        stride *= static_cast<uint64_t>(inputShape.GetDim(idx));
    }

    uint64_t outStride = 1;
    for (size_t rev = 0; rev < dimNum; ++rev) {
        const size_t idx = dimNum - 1 - rev;
        yStride[idx] = outStride;
        outStride *= static_cast<uint64_t>(yShape.GetDim(idx));

        if (idx < inputOffset) {
            inputStride[idx] = 0;
            continue;
        }

        const size_t inputIdx = idx - inputOffset;
        const int64_t inputDim = inputShape.GetDim(inputIdx);
        const int64_t outDim = yShape.GetDim(idx);
        inputStride[idx] = (inputDim == 1 && outDim != 1) ? 0 : baseStride[inputIdx];
    }
}

bool EncodeDataType(ge::DataType dataType, uint32_t& code)
{
    switch (dataType) {
        case ge::DT_FLOAT16: code = 0; return true;
        case ge::DT_BF16:    code = 1; return true;
        case ge::DT_FLOAT:   code = 2; return true;
        case ge::DT_INT32:   code = 3; return true;
        case ge::DT_INT16:   code = 4; return true;
        case ge::DT_UINT8:   code = 5; return true;
        case ge::DT_INT8:    code = 6; return true;
        default: return false;
    }
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* x1StorageShape = context->GetInputShape(0);
    const gert::StorageShape* x2StorageShape = context->GetInputShape(1);
    const auto* x1Desc = context->GetInputDesc(0);
    const auto* x2Desc = context->GetInputDesc(1);
    if (x1StorageShape == nullptr || x2StorageShape == nullptr || x1Desc == nullptr || x2Desc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& x1Shape = x1StorageShape->GetStorageShape();
    const gert::Shape& x2Shape = x2StorageShape->GetStorageShape();
    gert::Shape yShape;
    if (!BuildBroadcastShape(x1Shape, x2Shape, yShape)) {
        return ge::GRAPH_FAILED;
    }

    // 1. 【优化点】：判断是否需要启动广播机制
    uint8_t isBroadcast = 0;
    if (x1Shape.GetDimNum() != x2Shape.GetDimNum()) {
        isBroadcast = 1;
    } else {
        for (size_t i = 0; i < x1Shape.GetDimNum(); ++i) {
            if (x1Shape.GetDim(i) != x2Shape.GetDim(i)) {
                isBroadcast = 1;
                break;
            }
        }
    }

    uint32_t dataType = 0;
    if (x1Desc->GetDataType() != x2Desc->GetDataType() || !EncodeDataType(x1Desc->GetDataType(), dataType)) {
        return ge::GRAPH_FAILED;
    }

    // 2. 【优化点】：根据数据类型动态计算最优的 UB Tile 大小
    uint32_t typeSize = 4;
    switch (x1Desc->GetDataType()) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:   typeSize = 2; break;
        case ge::DT_UINT8:
        case ge::DT_INT8:    typeSize = 1; break;
        default:             typeSize = 4; break;
    }

    // 预留安全合理的 UB 空间给 3 个 Queue (x1, x2 为输入类型，y 为 uint8_t 状态值)
    // 设定单次流水线总可用 UB 内存大约为 128KB (留足富余空间防止编译爆仓)
    constexpr uint32_t kSafeUbSize = 128 * 1024;
    constexpr uint32_t kBufferNum = 2; // 双缓冲
    uint32_t maxTileLength = kSafeUbSize / (kBufferNum * (2 * typeSize + 1));

    // Ascend C 极其注重 32 字节对齐，转换成元素数量对齐
    uint32_t alignElements = 32 / typeSize;
    uint32_t tileLength = (maxTileLength / alignElements) * alignElements;
    if (tileLength == 0) {
        tileLength = alignElements;
    }

    uint64_t yShapeArr[kMaxDims] = {0};
    uint64_t yStride[kMaxDims] = {0};
    uint64_t x1Stride[kMaxDims] = {0};
    uint64_t x2Stride[kMaxDims] = {0};
    for (size_t i = 0; i < yShape.GetDimNum(); ++i) {
        yShapeArr[i] = static_cast<uint64_t>(yShape.GetDim(i));
    }
    BuildStrides(x1Shape, yShape, x1Stride, yStride);
    BuildStrides(x2Shape, yShape, x2Stride, yStride);

    const uint64_t totalSize = ShapeSize(yShape);
    const uint32_t blockDim = static_cast<uint32_t>(
        totalSize < static_cast<uint64_t>(kDefaultBlockDim) ? totalSize : kDefaultBlockDim);

    TensorEqualTilingData tiling;
    tiling.set_totalSize(totalSize);
    tiling.set_dimNum(static_cast<uint32_t>(yShape.GetDimNum()));
    tiling.set_dataType(dataType);
    tiling.set_blockDim(blockDim == 0 ? 1 : blockDim);
    tiling.set_isBroadcast(isBroadcast);
    tiling.set_tileLength(tileLength);
    tiling.set_yShape(yShapeArr);
    tiling.set_yStride(yStride);
    tiling.set_x1Stride(x1Stride);
    tiling.set_x2Stride(x2Stride);

    context->SetBlockDim(blockDim == 0 ? 1 : blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* x2_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x1_shape == nullptr || x2_shape == nullptr || y_shape == nullptr) {
        return GRAPH_FAILED;
    }
    if (!BuildBroadcastShape(*x1_shape, *x2_shape, *y_shape)) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class TensorEqual : public OpDef {
public:
    explicit TensorEqual(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};
OP_ADD(TensorEqual);
}
}