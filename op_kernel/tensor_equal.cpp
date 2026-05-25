#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kFloat16 = 0;
constexpr uint32_t kBfloat16 = 1;
constexpr uint32_t kFloat32 = 2;
constexpr uint32_t kInt32 = 3;
constexpr uint32_t kInt16 = 4;
constexpr uint32_t kUint8 = 5;
constexpr uint32_t kInt8 = 6;

constexpr int32_t BUFFER_NUM = 2; // 双缓冲开启管道重叠
}

template<typename T>
class KernelTensorEqual {
public:
    __aicore__ inline KernelTensorEqual() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TensorEqualTilingData& tilingData) {
        this->totalSize = tilingData.totalSize;
        this->dimNum = tilingData.dimNum;
        this->isBroadcast = tilingData.isBroadcast;
        this->tileLength = tilingData.tileLength;

        // 缓存步长数组到 AICore 寄存器/私有栈空间，避免深度循环中频繁读取 Tiling 缓冲区
        for (uint32_t i = 0; i < this->dimNum; ++i) {
            this->yStride[i] = tilingData.yStride[i];
            this->x1Stride[i] = tilingData.x1Stride[i];
            this->x2Stride[i] = tilingData.x2Stride[i];
        }

        const uint32_t blockDim = tilingData.blockDim;
        const uint32_t blockIdx = GetBlockIdx();
        
        // 核心切分：按输出总元素量（totalSize）平分任务给各 AICore
        this->blockLength = (this->totalSize + blockDim - 1) / blockDim;
        this->startOffset = blockIdx * this->blockLength;
        if (this->startOffset + this->blockLength > this->totalSize) {
            this->blockLength = this->totalSize - this->startOffset;
        }

        // 建立微型 Tile 块级循环计划
        this->loopCount = this->blockLength / this->tileLength;
        this->tailLength = this->blockLength % this->tileLength;

        // 初始化 Global Memory 视图
        if (this->isBroadcast == 0) {
            // 【快路径】：无广播，各核心直接对齐并偏移到对应起始点进行大连续块操作
            x1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1) + this->startOffset, this->blockLength);
            x2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2) + this->startOffset, this->blockLength);
        } else {
            // 【广播安全路径】：由于要进行跨维寻址，指针需保持零基座（不作 startOffset 预偏移）
            x1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1), this->totalSize); 
            x2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2), this->totalSize);
        }
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y) + this->startOffset, this->blockLength);

        // 使用 Host 动态计算出绝对安全的弹性切片大小初始化队列
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(uint8_t));
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < this->loopCount; i++) {
            CopyIn(i, this->tileLength);
            Compute(this->tileLength);
            CopyOut(i, this->tileLength);
        }
        if (this->tailLength > 0) {
            CopyIn(this->loopCount, this->tailLength);
            Compute(this->tailLength);
            CopyOut(this->loopCount, this->tailLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
        LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();

        if (this->isBroadcast == 0) {
            // 快路径：直接采用硬件 DMA 搬运连续区间，逼近硬件理论带宽极限
            DataCopy(x1Local, x1Gm[progress * this->tileLength], length);
            DataCopy(x2Local, x2Gm[progress * this->tileLength], length);
        } else {
            // 广播安全路径：利用坐标换算，把离散的、被广播的 GM 数据“收集(Gather)”写入连续的 UB 空间
            uint64_t tileStartIdx = this->startOffset + progress * this->tileLength;
            
            for (uint32_t i = 0; i < length; ++i) {
                uint64_t globalIdx = tileStartIdx + i;
                uint64_t remain = globalIdx;
                uint64_t x1Offset = 0;
                uint64_t x2Offset = 0;
                
                for (uint32_t dim = 0; dim < this->dimNum; ++dim) {
                    const uint64_t coord = remain / this->yStride[dim];
                    remain = remain % this->yStride[dim];
                    x1Offset += coord * this->x1Stride[dim];
                    x2Offset += coord * this->x2Stride[dim];
                }
                // 高效单点收集进 UB
                x1Local.SetValue(i, x1Gm.GetValue(x1Offset));
                x2Local.SetValue(i, x2Gm.GetValue(x2Offset));
            }
        }

        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute(uint32_t length) {
        LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();

        // 极其优雅的统一：因为 CopyIn 已经将数据标准化为连续的 LocalTensor
        // 这里可以直接享用 100% 向量化指令，免去了由于广播带来的各种计算分支判断！
        Compare(yLocal, x1Local, x2Local, CMPMODE::EQ, length);

        outQueueY.EnQue<uint8_t>(yLocal);
        inQueueX1.FreeTensor(&x1Local);
        inQueueX2.FreeTensor(&x2Local);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        // 无论输入端如何广播，输出张量 y 的物理分布始终是完全连续的，因此写回一律走高性能 DataCopy
        DataCopy(yGm[progress * this->tileLength], yLocal, length);
        outQueueY.FreeTensor(&yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX1;
    TQue<QuePosition::VECIN, 1> inQueueX2;
    TQue<QuePosition::VECOUT, 1> outQueueY;

    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<uint8_t> yGm;

    uint64_t totalSize;
    uint32_t dimNum;
    uint64_t blockLength;
    uint64_t startOffset;
    uint32_t tileLength;
    int32_t loopCount;
    uint32_t tailLength;
    uint8_t isBroadcast;

    uint64_t yStride[25];
    uint64_t x1Stride[25];
    uint64_t x2Stride[25];
};

extern "C" __global__ __aicore__ void tensor_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    switch (tiling_data.dataType) {
        case kFloat16: {
            KernelTensorEqual<half> op;
            op.Init(x1, x2, y, tiling_data);
            op.Process();
            break;
        }
        case kBfloat16: {
            KernelTensorEqual<bfloat16_t> op;
            op.Init(x1, x2, y, tiling_data);
            op.Process();
            break;
        }
        case kFloat32: {
            KernelTensorEqual<float> op;
            op.Init(x1, x2, y, tiling_data);
            op.Process();
            break;
        }
        case kInt32: {
            KernelTensorEqual<int32_t> op;
            op.Init(x1, x2, y, tiling_data);
            op.Process();
            break;
        }
        case kInt16: {
            KernelTensorEqual<int16_t> op;
            op.Init(x1, x2, y, tiling_data);
            op.Process();
            break;
        }
        case kUint8: {
            KernelTensorEqual<uint8_t> op;
            op.Init(x1, x2, y, tiling_data);
            op.Process();
            break;
        }
        case kInt8: {
            KernelTensorEqual<int8_t> op;
            op.Init(x1, x2, y, tiling_data);
            op.Process();
            break;
        }
        default:
            break;
    }
}