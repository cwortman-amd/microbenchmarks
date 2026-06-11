#pragma once

#include "kittens.cuh"

// Contract for the next LDS transpose-read experiment.
//
// This header is intentionally not included by kernel.cpp yet. It records the
// concrete HipKittens layout types that should be used for a GEMM-only probe
// before the layout is ported into fused AG+MM.
//
// Current fused AG+MM path:
//   A: rt_bf<32,64,row_l,rt_16x32_s>
//   B: rt_bf<32,64,row_l,rt_16x32_s>
//   C: rt_fl<32,32,col_l,rt_16x16_s>
//   mma_ABt(C, A, B, C)
//
// Transpose-read probe path:
//   A: rt_bf<64,32,col_l,rt_16x32_s>  // shared_to_register emits ds_read_b64_tr_b16
//   B: rt_bf<64,32,col_l,rt_16x32_s>  // shared_to_register emits ds_read_b64_tr_b16
//   C: rt_fl<32,32,col_l,rt_16x16_s>
//   mma_AtB(C, A, B, C)
//
// The real probe should:
//   1. Stage a local GEMM A/B tile into LDS using HK global-to-shared paths.
//   2. Load the shared tiles into the column-layout register tiles below.
//   3. Call mma_AtB, not mma_ABt.
//   4. Store C through a layout-correct global store path.
//   5. Compare against the current row-layout GEMM/AG+MM output and inspect
//      compiler resources before porting the same layout into the Iris kernel.

namespace hk_gemm_layout_probe {

constexpr int BLOCK_SIZE = 64;
constexpr int HALF_BLOCK_SIZE = BLOCK_SIZE / 2;

using RowA = kittens::rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, kittens::row_l, kittens::rt_16x32_s>;
using RowB = kittens::rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, kittens::row_l, kittens::rt_16x32_s>;
using ColA = kittens::rt_bf<BLOCK_SIZE, HALF_BLOCK_SIZE, kittens::col_l, kittens::rt_16x32_s>;
using ColB = kittens::rt_bf<BLOCK_SIZE, HALF_BLOCK_SIZE, kittens::col_l, kittens::rt_16x32_s>;
using Accum = kittens::rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, kittens::col_l, kittens::rt_16x16_s>;

static_assert(Accum::rows == ColA::cols);
static_assert(Accum::cols == ColB::cols);
static_assert(ColA::rows == ColB::rows);

} // namespace hk_gemm_layout_probe
