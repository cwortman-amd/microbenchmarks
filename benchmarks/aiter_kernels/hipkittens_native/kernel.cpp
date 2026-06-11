#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

constexpr int BLOCK_SIZE = 64;
constexpr int M_BLOCK = 2;
constexpr int N_BLOCK = 4;
constexpr int HALF_BLOCK_SIZE = BLOCK_SIZE / 2;
constexpr int NEW_ROW_BLOCK_SIZE = BLOCK_SIZE * M_BLOCK;
constexpr int NEW_COL_BLOCK_SIZE = BLOCK_SIZE * N_BLOCK;

#define NUM_PRODUCER_WORKERS (4)
#define NUM_CONSUMER_WORKERS (M_BLOCK * 4)
#define NUM_THREADS ((NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS) * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)

using G = kittens::group<NUM_PRODUCER_WORKERS>;
using A_slice = rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s>;
using B_slice = rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s>;

struct iris_context_view {
    int64_t* data;

    __device__ int cur_rank() const { return static_cast<int>(data[0]); }
    __device__ int world_size() const { return static_cast<int>(data[1]); }
    __device__ uintptr_t get_heap_base(int rank) const { return static_cast<uintptr_t>(data[2 + rank]); }

    template <typename T>
    __device__ T* translate(const T* ptr, int remote_rank) const {
        const uintptr_t cur_base = get_heap_base(cur_rank());
        const uintptr_t remote_base = get_heap_base(remote_rank);
        const uintptr_t offset = reinterpret_cast<uintptr_t>(ptr) - cur_base;
        return reinterpret_cast<T*>(remote_base + offset);
    }

    template <typename T>
    __device__ void store(T* ptr, T value, int remote_rank) const {
        *translate(ptr, remote_rank) = value;
    }
};

template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void iris_store_tile(
    const GL &dst,
    const RT &src,
    const COORD &idx,
    const iris_context_view& iris_ctx,
    int remote_rank
) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    const int laneid = kittens::laneid();
    const int row_offset = src.base_tile_stride * (laneid / src.base_tile_cols);
    const int col_offset = laneid % src.base_tile_cols;

    #pragma unroll
    for (int i = 0; i < src.height; i++) {
        #pragma unroll
        for (int j = 0; j < src.width; j++) {
            const int col = j * src.base_tile_cols + col_offset;
            #pragma unroll
            for (int k = 0; k < src.base_tile_num_strides; k++) {
                int row = i * src.base_tile_rows + row_offset + k * src.base_tile_elements_per_stride_group;
                #pragma unroll
                for (int l = 0; l < src.base_tile_stride / packing; l++) {
                    int src_idx = l + k * src.base_tile_stride / packing;
                    U val_x = base_types::convertor<U, T>::convert(src.tiles[i][j].data[src_idx].x);
                    U val_y = base_types::convertor<U, T>::convert(src.tiles[i][j].data[src_idx].y);
                    iris_ctx.store(&dst_ptr[(row + l * 2) * row_stride + col], val_x, remote_rank);
                    iris_ctx.store(&dst_ptr[(row + l * 2 + 1) * row_stride + col], val_y, remote_rank);
                }
            }
        }
    }
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void iris_store_tile(const GL &dst, const RT &src, const COORD &idx, const iris_context_view& iris_ctx) {
    iris_store_tile<2, RT, GL, COORD>(dst, src, idx, iris_ctx, iris_ctx.cur_rank());
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void iris_store_tile(const GL &dst, const RT &src, const COORD &idx, const iris_context_view& iris_ctx, int remote_rank) {
    iris_store_tile<2, RT, GL, COORD>(dst, src, idx, iris_ctx, remote_rank);
}

struct ag_mm_globals {
    gl<bf16, -1, -1, -1, -1> a;
    gl<bf16, -1, -1, -1, -1> b_t;
    gl<bf16, -1, -1, -1, -1> c;
    uint64_t iris_context_ptr;
    int M_total;
    int M_local;
    int N;
    int K;
    int row_offset;

    dim3 grid()  { return dim3(ceil_div(N, NEW_COL_BLOCK_SIZE), ceil_div(M_total, NEW_ROW_BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 98304; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void ag_mm_local_tk(ag_mm_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    iris_context_view iris_ctx{reinterpret_cast<int64_t*>(g.iris_context_ptr)};

    using ST_A = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    ST_A (&As)[2][M_BLOCK][2] = al.allocate<ST_A, 2, M_BLOCK, 2>();
    ST_B (&Bs)[2][N_BLOCK][2] = al.allocate<ST_B, 2, N_BLOCK, 2>();
    rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, col_l, rt_16x16_s> C_accum[2][2];

    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    const int WGM = 4;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM * WGM);
    const int num_pid_m = ceil_div(g.M_total, NEW_ROW_BLOCK_SIZE);
    const int num_pid_n = ceil_div(g.N, NEW_COL_BLOCK_SIZE);
    const int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    int row = pid_m * M_BLOCK;
    int col = pid_n * N_BLOCK;

    // AG+MM ownership: A is sharded by M across ranks in symmetric memory. Each
    // CTA computes a global M tile, translates A's base pointer to the owning
    // rank's heap, then reuses HK's coalesced global-to-LDS path unchanged.
    const int global_row_start = row * BLOCK_SIZE;
    const int source_rank = min(global_row_start / g.M_local, iris_ctx.world_size() - 1);
    const int source_rank_row0 = source_rank * g.M_local;
    const int local_row = (global_row_start - source_rank_row0) / BLOCK_SIZE;
    auto a_src = g.a;
    using AType = typename decltype(g.a)::dtype;
    const uintptr_t cur_heap_base = iris_ctx.get_heap_base(iris_ctx.cur_rank());
    const uintptr_t src_heap_base = iris_ctx.get_heap_base(source_rank);
    const uintptr_t a_offset = reinterpret_cast<uintptr_t>(g.a.raw_ptr) - cur_heap_base;
    a_src.raw_ptr = reinterpret_cast<AType*>(src_heap_base + a_offset);

    int warp_id = kittens::warpid();
    int local_warp_id = warp_id % 4;
    int warp_group_id = warp_id / 4;
    bool is_producer = (warp_group_id == 0);
    bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id - 1 : 0;

    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>::dtype;
    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0][0], g.b_t, swizzled_offsets_B);

    int tic = 0;
    int toc = 1;
    if (is_producer) {
        #pragma unroll
        for (int m = 0; m < M_BLOCK; m++) {
            G::load<2, false>(As[tic][m][0], a_src, {0, 0, local_row * 2 + 2 * m + 0, 0}, swizzled_offsets_A);
            G::load<2, false>(As[tic][m][1], a_src, {0, 0, local_row * 2 + 2 * m + 1, 0}, swizzled_offsets_A);
        }
        #pragma unroll
        for (int n = 0; n < N_BLOCK; n++) {
            G::load<2, false>(Bs[tic][n][0], g.b_t, {0, 0, col * 2 + 2 * n + 0, 0}, swizzled_offsets_B);
            G::load<2, false>(Bs[tic][n][1], g.b_t, {0, 0, col * 2 + 2 * n + 1, 0}, swizzled_offsets_B);
        }
        __builtin_amdgcn_s_waitcnt(0);
    }
    __syncthreads();

    if (is_consumer) {
        zero(C_accum[0][0]);
        zero(C_accum[0][1]);
        zero(C_accum[1][0]);
        zero(C_accum[1][1]);
    }

    int num_tiles = g.K / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles - 1; ++tile, tic ^= 1, toc ^= 1) {
        if (is_producer) {
            #pragma unroll
            for (int m = 0; m < M_BLOCK; m++) {
                G::load<2, false>(As[toc][m][0], a_src, {0, 0, local_row * 2 + 2 * m + 0, tile + 1}, swizzled_offsets_A);
                G::load<2, false>(As[toc][m][1], a_src, {0, 0, local_row * 2 + 2 * m + 1, tile + 1}, swizzled_offsets_A);
            }
            #pragma unroll
            for (int n = 0; n < N_BLOCK; n++) {
                G::load<2, false>(Bs[toc][n][0], g.b_t, {0, 0, col * 2 + 2 * n + 0, tile + 1}, swizzled_offsets_B);
                G::load<2, false>(Bs[toc][n][1], g.b_t, {0, 0, col * 2 + 2 * n + 1, tile + 1}, swizzled_offsets_B);
            }
            __builtin_amdgcn_s_waitcnt(0);
        } else if (is_consumer) {
            A_slice a0;
            B_slice b0, b1;
            auto st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][0], {0, 0});
            load(b0, st_subtile_b);
            auto st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][0], {0, 0});
            load(a0, st_subtile_a);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[0][0], a0, b0, C_accum[0][0]);
            __builtin_amdgcn_s_setprio(0);

            st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][1], {0, 0});
            load(b1, st_subtile_b);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[0][1], a0, b1, C_accum[0][1]);
            __builtin_amdgcn_s_setprio(0);

            st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][1], {0, 0});
            load(a0, st_subtile_a);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[1][0], a0, b0, C_accum[1][0]);
            mma_ABt(C_accum[1][1], a0, b1, C_accum[1][1]);
            __builtin_amdgcn_s_setprio(0);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    if (is_consumer) {
        A_slice a0;
        B_slice b0, b1;
        auto st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][0], {0, 0});
        load(b0, st_subtile_b);
        auto st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][0], {0, 0});
        load(a0, st_subtile_a);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][0], a0, b0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);

        st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][1], {0, 0});
        load(b1, st_subtile_b);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][1], a0, b1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);

        st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][1], {0, 0});
        load(a0, st_subtile_a);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[1][0], a0, b0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], a0, b1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
    }

    if (is_consumer) {
        iris_store_tile(g.c, C_accum[0][0], {0, 0, (row + consumer_idx) * 2 + 0, (col + local_warp_id) * 2 + 0}, iris_ctx);
        iris_store_tile(g.c, C_accum[0][1], {0, 0, (row + consumer_idx) * 2 + 0, (col + local_warp_id) * 2 + 1}, iris_ctx);
        iris_store_tile(g.c, C_accum[1][0], {0, 0, (row + consumer_idx) * 2 + 1, (col + local_warp_id) * 2 + 0}, iris_ctx);
        iris_store_tile(g.c, C_accum[1][1], {0, 0, (row + consumer_idx) * 2 + 1, (col + local_warp_id) * 2 + 1}, iris_ctx);
    }
}

void dispatch_ag_mm(ag_mm_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)ag_mm_local_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    ag_mm_local_tk<<<g.grid(), g.block(), mem_size>>>(g);
}

struct mm_rs_write_globals {
    gl<bf16, -1, -1, -1, -1> a;
    gl<bf16, -1, -1, -1, -1> b_t;
    gl<bf16, -1, -1, -1, -1> scratch;
    uint64_t iris_context_ptr;
    int M_total;
    int M_shard;
    int N;
    int K;

    dim3 grid()  { return dim3(ceil_div(N, NEW_COL_BLOCK_SIZE), ceil_div(M_total, NEW_ROW_BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 98304; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void mm_rs_write_tk(mm_rs_write_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    iris_context_view iris_ctx{reinterpret_cast<int64_t*>(g.iris_context_ptr)};

    using ST_A = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    ST_A (&As)[2][M_BLOCK][2] = al.allocate<ST_A, 2, M_BLOCK, 2>();
    ST_B (&Bs)[2][N_BLOCK][2] = al.allocate<ST_B, 2, N_BLOCK, 2>();
    rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, col_l, rt_16x16_s> C_accum[2][2];

    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    const int WGM = 4;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM * WGM);
    const int num_pid_m = ceil_div(g.M_total, NEW_ROW_BLOCK_SIZE);
    const int num_pid_n = ceil_div(g.N, NEW_COL_BLOCK_SIZE);
    const int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    int row = pid_m * M_BLOCK;
    int col = pid_n * N_BLOCK;

    const int global_row_start = row * BLOCK_SIZE;
    const int dest_rank = min(global_row_start / g.M_shard, iris_ctx.world_size() - 1);
    const int dest_rank_row0 = dest_rank * g.M_shard;
    const int local_row = (global_row_start - dest_rank_row0) / BLOCK_SIZE;

    int warp_id = kittens::warpid();
    int local_warp_id = warp_id % 4;
    int warp_group_id = warp_id / 4;
    bool is_producer = (warp_group_id == 0);
    bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id - 1 : 0;

    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>::dtype;
    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0][0], g.b_t, swizzled_offsets_B);

    int tic = 0;
    int toc = 1;
    if (is_producer) {
        #pragma unroll
        for (int m = 0; m < M_BLOCK; m++) {
            G::load<2, false>(As[tic][m][0], g.a, {0, 0, row * 2 + 2 * m + 0, 0}, swizzled_offsets_A);
            G::load<2, false>(As[tic][m][1], g.a, {0, 0, row * 2 + 2 * m + 1, 0}, swizzled_offsets_A);
        }
        #pragma unroll
        for (int n = 0; n < N_BLOCK; n++) {
            G::load<2, false>(Bs[tic][n][0], g.b_t, {0, 0, col * 2 + 2 * n + 0, 0}, swizzled_offsets_B);
            G::load<2, false>(Bs[tic][n][1], g.b_t, {0, 0, col * 2 + 2 * n + 1, 0}, swizzled_offsets_B);
        }
        __builtin_amdgcn_s_waitcnt(0);
    }
    __syncthreads();

    if (is_consumer) {
        zero(C_accum[0][0]);
        zero(C_accum[0][1]);
        zero(C_accum[1][0]);
        zero(C_accum[1][1]);
    }

    int num_tiles = g.K / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles - 1; ++tile, tic ^= 1, toc ^= 1) {
        if (is_producer) {
            #pragma unroll
            for (int m = 0; m < M_BLOCK; m++) {
                G::load<2, false>(As[toc][m][0], g.a, {0, 0, row * 2 + 2 * m + 0, tile + 1}, swizzled_offsets_A);
                G::load<2, false>(As[toc][m][1], g.a, {0, 0, row * 2 + 2 * m + 1, tile + 1}, swizzled_offsets_A);
            }
            #pragma unroll
            for (int n = 0; n < N_BLOCK; n++) {
                G::load<2, false>(Bs[toc][n][0], g.b_t, {0, 0, col * 2 + 2 * n + 0, tile + 1}, swizzled_offsets_B);
                G::load<2, false>(Bs[toc][n][1], g.b_t, {0, 0, col * 2 + 2 * n + 1, tile + 1}, swizzled_offsets_B);
            }
            __builtin_amdgcn_s_waitcnt(0);
        } else if (is_consumer) {
            A_slice a0;
            B_slice b0, b1;
            auto st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][0], {0, 0});
            load(b0, st_subtile_b);
            auto st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][0], {0, 0});
            load(a0, st_subtile_a);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[0][0], a0, b0, C_accum[0][0]);
            __builtin_amdgcn_s_setprio(0);

            st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][1], {0, 0});
            load(b1, st_subtile_b);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[0][1], a0, b1, C_accum[0][1]);
            __builtin_amdgcn_s_setprio(0);

            st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][1], {0, 0});
            load(a0, st_subtile_a);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[1][0], a0, b0, C_accum[1][0]);
            mma_ABt(C_accum[1][1], a0, b1, C_accum[1][1]);
            __builtin_amdgcn_s_setprio(0);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    if (is_consumer) {
        A_slice a0;
        B_slice b0, b1;
        auto st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][0], {0, 0});
        load(b0, st_subtile_b);
        auto st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][0], {0, 0});
        load(a0, st_subtile_a);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][0], a0, b0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);

        st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][1], {0, 0});
        load(b1, st_subtile_b);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][1], a0, b1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);

        st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][1], {0, 0});
        load(a0, st_subtile_a);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[1][0], a0, b0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], a0, b1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
    }

    if (is_consumer) {
        const int slot = iris_ctx.cur_rank();
        iris_store_tile(g.scratch, C_accum[0][0], {0, slot, (local_row + consumer_idx) * 2 + 0, (col + local_warp_id) * 2 + 0}, iris_ctx, dest_rank);
        iris_store_tile(g.scratch, C_accum[0][1], {0, slot, (local_row + consumer_idx) * 2 + 0, (col + local_warp_id) * 2 + 1}, iris_ctx, dest_rank);
        iris_store_tile(g.scratch, C_accum[1][0], {0, slot, (local_row + consumer_idx) * 2 + 1, (col + local_warp_id) * 2 + 0}, iris_ctx, dest_rank);
        iris_store_tile(g.scratch, C_accum[1][1], {0, slot, (local_row + consumer_idx) * 2 + 1, (col + local_warp_id) * 2 + 1}, iris_ctx, dest_rank);
    }
}

void dispatch_mm_rs_write(mm_rs_write_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)mm_rs_write_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    mm_rs_write_tk<<<g.grid(), g.block(), mem_size>>>(g);
}

struct mm_rs_reduce_globals {
    gl<bf16, -1, -1, -1, -1> scratch;
    gl<bf16, -1, -1, -1, -1> y;
    int M_shard;
    int N;
    int world;
    float scale;

    dim3 grid()  { return dim3(ceil_div(M_shard * N, 256)); }
    dim3 block() { return dim3(256); }
};

__global__ void mm_rs_reduce_kernel(mm_rs_reduce_globals g) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = g.M_shard * g.N;
    if (idx >= total) {
        return;
    }
    int row = idx / g.N;
    int col = idx - row * g.N;
    float acc = 0.0f;
    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        if (slot < g.world) {
            bf16 v = g.scratch.raw_ptr[((slot * g.M_shard + row) * g.N) + col];
            acc += base_types::convertor<float, bf16>::convert(v);
        }
    }
    g.y.raw_ptr[idx] = base_types::convertor<bf16, float>::convert(acc * g.scale);
}

void dispatch_mm_rs_reduce(mm_rs_reduce_globals g) {
    mm_rs_reduce_kernel<<<g.grid(), g.block()>>>(g);
}

PYBIND11_MODULE(hk_iris_fused, m) {
    m.doc() = "HipKittens + Iris experimental fused collective backend";
    py::bind_function<dispatch_ag_mm>(m, "dispatch_ag_mm",
        &ag_mm_globals::a,
        &ag_mm_globals::b_t,
        &ag_mm_globals::c,
        &ag_mm_globals::iris_context_ptr,
        &ag_mm_globals::M_total,
        &ag_mm_globals::M_local,
        &ag_mm_globals::N,
        &ag_mm_globals::K,
        &ag_mm_globals::row_offset
    );
    py::bind_function<dispatch_mm_rs_write>(m, "dispatch_mm_rs_write",
        &mm_rs_write_globals::a,
        &mm_rs_write_globals::b_t,
        &mm_rs_write_globals::scratch,
        &mm_rs_write_globals::iris_context_ptr,
        &mm_rs_write_globals::M_total,
        &mm_rs_write_globals::M_shard,
        &mm_rs_write_globals::N,
        &mm_rs_write_globals::K
    );
    py::bind_function<dispatch_mm_rs_reduce>(m, "dispatch_mm_rs_reduce",
        &mm_rs_reduce_globals::scratch,
        &mm_rs_reduce_globals::y,
        &mm_rs_reduce_globals::M_shard,
        &mm_rs_reduce_globals::N,
        &mm_rs_reduce_globals::world,
        &mm_rs_reduce_globals::scale
    );
}

