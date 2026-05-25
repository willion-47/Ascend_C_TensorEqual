# Ascend_C_TensorEqual

基于 Ascend C 和华为 CANN 架构开发的 TensorEqual（张量绝对差值）自定义算子。针对 Orange Pi AI Pro（昇腾芯片）进行了多核与双缓冲队列优化，实现 $z = |x - y|$ 的高性能端侧计算。

# TensorEqual Custom Operator for CANN

[![CANN Version](https://img.shields.io/badge/CANN-7.0%2B-blue.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-Orange%20Pi%20AI%20Pro-orange.svg)]()
[![Language](https://img.shields.io/badge/Language-Ascend%20C-green.svg)]()

## 📖 简介 (Introduction)

本项目提供了一个基于华为 CANN 架构和 Ascend C 语言开发的 **TensorEqual** 自定义算子。已在搭载昇腾 AI 处理器的 **Orange Pi AI Pro** 开发板上完成功能验证与性能测试。

该算子的核心逻辑是计算两个相同形状张量的**元素级绝对差值 (Absolute Difference)**。
其数学表达式为：
$$z_{i} = |x_{i} - y_{i}|$$

### ✨ 性能优化亮点
* **多核负载均衡 (Multi-core Tiling):** 在 Host 侧基于实际数据量和硬件 AI Core 数量动态切分任务，确保多核利用率最大化。
* **内存对齐 (Memory Alignment):** 严格遵循 32-Byte 内存对齐原则（单次处理 8 个 Float32 元素），提升 Vector 单元的访存效率。
* **双缓冲流水线 (Double Buffering):** Kernel 侧使用深度为 2 的 TQue 队列，实现了搬入（CopyIn）、计算（Compute）和搬出（CopyOut）的并行流水线作业。
* **In-place 内存复用:** 在计算步骤中巧妙复用临时缓冲区，规避了原地修改引发的访存冲突。

## ⚙️ 环境要求 (Prerequisites)

- **硬件:** Orange Pi AI Pro (Ascend 310B / 910B 兼容)
- **操作系统:** Ubuntu 22.04 / openEuler 22.03
- **依赖软件:**
  - CANN Toolkit (推荐 7.0.0 及以上)
  - CMake >= 3.16
  - gcc/g++ >= 9.4.0

## 📐 算子规格 (Operator Specification)

| 算子类型 (OpType) | `TensorEqual` |
| :--- | :--- |
| **输入 `x`** | 数据类型: `float32` (DT_FLOAT) <br> 格式: `ND` |
| **输入 `y`** | 数据类型: `float32` (DT_FLOAT) <br> 格式: `ND` |
| **输出 `z`** | 数据类型: `float32` (DT_FLOAT) <br> 格式: `ND` |

> **注意:** `x` 和 `y` 必须具有完全相同的 Shape 和数据类型。

## 📂 核心代码结构 (Directory Structure)

```text
├── op_host
│   ├── tensor_equal.cpp       # Host侧算子原型注册与多核Tiling切分实现
│   └── tensor_equal_tiling.h  # Tiling 数据结构定义
├── op_kernel
│   └── tensor_equal.cpp       # Device侧 Ascend C Kernel 具体计算实现
