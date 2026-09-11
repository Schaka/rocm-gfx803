// Multi-token GEMV for gfx803 (Polaris/GCN3) decode, 2 <= M <= 16.
//
// Why this kernel exists: decode at M > 1 has no fast path on this arch. The
// hand-written tiled GEMM that serves prefill tiles 64 activation rows at a
// time, so at M=8 it computes 64 rows of MACs to produce 8 rows of output --
// an 8x waste that puts it in the compute-bound regime instead of the
// memory-bound one. Measured, it moved 12-48 GB/s of the card's ~168 GB/s
// copy bandwidth at these sizes, and decode spends that on 113 of the 114
// weight reads a step makes. rocBLAS's hand-written GEMV (ops.LLMM1) does
// reach high bandwidth, but it asserts N == 1: its lanes walk K and each
// lane's accumulator holds one row, so a second token would need a second
// lane group while its grid is already spent on the feature axis.
//
// This kernel is that missing lane group, kept in registers instead. A wave
// of 64 lanes is divided between output rows and lanes-along-K:
//
//     lane = lrow * LPR + klane         (klane walks K, lrow picks a row)
//
// and every lane keeps one fp32 accumulator per token. The weight chunk a
// lane loads is then reused for the whole batch, which is the point: at M=8
// it serves eight tokens' worth of MACs, against one for a layout that puts
// the tokens in the lanes.
//
// The tuning is all one trade, and it is not about arithmetic. Saturation
// needs weight loads in flight, which needs waves per CU, which the register
// count sets: MTOK = 2/4/8/16 compile to 30/46/62/102 VGPRs, and the same
// weight shape moved 181 GB/s at the first count and 95 GB/s at the third.
// So the batch costs registers, and the batch is what buys reuse. The
// measured optimum divides the wave as LPR=32 lanes per row -- two rows per
// wave -- with the batch held in accumulators.
//
// Two variants are worth not re-attempting. Putting the tokens in the lanes
// (three-way split of the wave, one accumulator per lane) holds 28 VGPRs
// flat across every batch size and measured slower everywhere, because the
// reuse then has to come from cache rather than registers. Summing the eight
// products of a chunk in packed fp16 instead of converting to fp32 was also
// measured, at M=2..8, and came out no faster (46-76 GB/s against 58-83
// GB/s for fp32 accumulation at M=8), so it was not worth its rounding.
//
// Weights are read in their native [N, K] layout. That is deliberate: the
// [K, N] layout the tiled GEMM wants has to be produced by a transposed copy
// that costs VRAM and caps which weights are eligible (lm_head, a quarter of
// this model's weights, is past the cap and falls back to Tensile). Here the
// lanes are adjacent along K, so the native row-major read is already
// coalesced and no weight is out of reach.
//
// The dispatch instantiates the next power of two at or above M, so the
// kernel has to mask the slots past the real batch: an unmasked store writes
// into whatever follows the output tensor. The exact width gets its own
// instantiation because the mask costs about a tenth of the throughput at
// M=8 when it is only ever true.
//
// Compile (from the directory holding the Python loader, which is where the
// .so must land for its __file__-relative ctypes path to find it):
//   hipcc --offload-arch=gfx803 -O3 -shared -fPIC \
//     -o libgfx803gemv_m.so ../../gfx803_kernels/gfx803_gemv_m.hip
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WAVE 64

// Tunables, overridable from the compile line because which values this card
// wants is not visible from the source. LPR is how many lanes share the K
// walk of one row; the rest of the wave covers further rows. The launch bound
// is the other half of the occupancy trade: it decides how many waves a CU
// holds, and therefore how much weight traffic can be in flight.
#ifndef GEMV_M_LPR
#define GEMV_M_LPR 32
#endif
#ifndef GEMV_M_THREADS
#define GEMV_M_THREADS 256
#endif
#ifndef GEMV_M_MIN_BLOCKS
#define GEMV_M_MIN_BLOCKS 2
#endif
#ifndef GEMV_M_UNROLL
#define GEMV_M_UNROLL 2
#endif

// Eight halves, kept packed. Only the tail of a row can clamp a chunk, and it
// is reached only when K is not a multiple of 8.
__device__ __forceinline__ void load8_packed(const half* __restrict__ p,
                                             bool full, int K, int k,
                                             half2* out)
{
    if (full) {
        *reinterpret_cast<float4*>(out) = *reinterpret_cast<const float4*>(p);
    } else {
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const half lo = (k + 2 * e < K) ? p[2 * e] : __float2half(0.f);
            const half hi = (k + 2 * e + 1 < K) ? p[2 * e + 1] : __float2half(0.f);
            out[e] = __halves2half2(lo, hi);
        }
    }
}

__device__ __forceinline__ void cvt8(const half2* __restrict__ p, float* out)
{
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const float2 v = __half22float2(p[j]);
        out[2 * j] = v.x;
        out[2 * j + 1] = v.y;
    }
}

// EXACT says the batch is the whole template width, which lets the masks in
// the token loop fold away. The dispatch cannot know that at compile time, so
// the exact width gets its own instantiation: masking costs about a tenth of
// the throughput at M=8 when it is only ever true.
template <int MTOK, int LPR, bool EXACT>
__global__ __launch_bounds__(GEMV_M_THREADS, GEMV_M_MIN_BLOCKS)
void gfx803_gemv_m_kernel(const half* __restrict__ X,
                          const half* __restrict__ W, half* __restrict__ C,
                          int M, int N, int K)
{
    static_assert(LPR <= WAVE && WAVE % LPR == 0, "LPR must divide the wave");
    constexpr int ROWS_PER_WAVE = WAVE / LPR;
    constexpr int WAVES_PER_BLOCK = GEMV_M_THREADS / WAVE;

    const int lane = threadIdx.x & (WAVE - 1);
    const int wave = threadIdx.x / WAVE;
    // Lane within its K group, and which of the wave's rows it accumulates.
    const int klane = lane % LPR;
    const int lrow = lane / LPR;
    const int rows_per_block = ROWS_PER_WAVE * WAVES_PER_BLOCK;
    const int64_t row =
        (int64_t)blockIdx.x * rows_per_block + wave * ROWS_PER_WAVE + lrow;
    const bool row_ok = (row < N);
    const half* __restrict__ wrow = W + row * K;

    float acc[MTOK];
    #pragma unroll
    for (int m = 0; m < MTOK; m++) acc[m] = 0.0f;

    const int chunks = (K + 7) >> 3;
    const int niter = (chunks + LPR - 1) / LPR;

    #pragma unroll GEMV_M_UNROLL
    for (int i = 0; i < niter; i++) {
        const int c = klane + i * LPR;
        // K is normally a multiple of the chunk size, so this only rejects
        // anything in the last iteration and only for the high lanes of a
        // group. It stays a predicate rather than a break so the loop keeps a
        // trip count the compiler can unroll.
        if (c < chunks) {
            const int k = c * 8;
            const bool full = (k + 8 <= K);
            // The batch's activation is staged, and the weight chunk is
            // loaded with those loads still in flight. Both orderings
            // compile to the same register count, so the compiler is not
            // choosing between them on pressure: staging first and putting
            // the weight load last measured 8% faster at M=8 and 28% at M=2
            // than loading the weight first. Each lane's weight chunk is then
            // reused for every staged token, which is where the batch's
            // bytes-per-flop comes from -- the variant that moved the tokens
            // into the lanes instead, so that a chunk served one token per
            // lane, held a flat 28 VGPRs and was slower at every batch size.
            half2 x2[MTOK][4];
            #pragma unroll
            for (int m = 0; m < MTOK; m++) {
                if (EXACT || m < M)
                    load8_packed(X + (int64_t)m * K + k, full, K, k, x2[m]);
                else
                    #pragma unroll
                    for (int e = 0; e < 4; e++)
                        x2[m][e] = __floats2half2_rn(0.f, 0.f);
            }
            float wv[8];
            if (row_ok) {
                half2 w2[4];
                load8_packed(wrow + k, full, K, k, w2);
                cvt8(w2, wv);
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++) wv[e] = 0.f;
            }
            #pragma unroll
            for (int m = 0; m < MTOK; m++) {
                float xv[8];
                cvt8(x2[m], xv);
                #pragma unroll
                for (int e = 0; e < 8; e++) acc[m] = fmaf(xv[e], wv[e], acc[m]);
            }
        }
    }

    // Each lane holds a partial over the chunks its group walked; a row needs
    // the whole group's partials. xor-shuffles stay inside one group because
    // the groups are aligned to LPR and LPR is a power of two.
    #pragma unroll
    for (int m = 0; m < MTOK; m++) {
        #pragma unroll
        for (int mask = LPR / 2; mask >= 1; mask /= 2)
            acc[m] += __shfl_xor(acc[m], mask);
    }

    if (klane == 0 && row_ok) {
        #pragma unroll
        for (int m = 0; m < MTOK; m++)
            if (EXACT || m < M) C[(int64_t)m * N + row] = __float2half(acc[m]);
    }
}

extern "C" {

// One instantiation of the kernel, including the grid its wave layout
// implies: rows per block is how many rows a wave owns times how many waves
// a block has.
#define LAUNCH_GEMV_M(MTOK)                                                    \
    do {                                                                       \
        constexpr int RPB = (WAVE / GEMV_M_LPR) * (GEMV_M_THREADS / WAVE);     \
        dim3 grid((N + RPB - 1) / RPB);                                        \
        if (M == MTOK)                                                         \
            gfx803_gemv_m_kernel<MTOK, GEMV_M_LPR, true>                       \
                <<<grid, GEMV_M_THREADS, 0, s>>>(x, w, c, M, N, K);            \
        else                                                                   \
            gfx803_gemv_m_kernel<MTOK, GEMV_M_LPR, false>                      \
                <<<grid, GEMV_M_THREADS, 0, s>>>(x, w, c, M, N, K);            \
    } while (0)

// C[M, N] = X[M, K] @ W[N, K]^T, all row-major fp16, W in vLLM's native
// [out_features, in_features] layout. M must be in [2, 16]; larger M stays on
// the tiled GEMM, whose 64-row tile is full at those sizes.
void gfx803_gemv_m_launch(const void* X, const void* W, void* C, int M, int N,
                          int K, void* stream)
{
    if (M < 2 || M > 16) return;
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    const half* x = reinterpret_cast<const half*>(X);
    const half* w = reinterpret_cast<const half*>(W);
    half* c = reinterpret_cast<half*>(C);
    if (M <= 2) LAUNCH_GEMV_M(2);
    else if (M <= 4) LAUNCH_GEMV_M(4);
    else if (M <= 8) LAUNCH_GEMV_M(8);
    else LAUNCH_GEMV_M(16);
}

} // extern "C"
