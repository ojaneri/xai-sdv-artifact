/*
 * Explanation methods for the tabular IDS model, in portable C.
 *
 * The same file builds for the host (host_check.c, validated against the
 * `shap` package) and for Cortex-M targets (xai_fw.c). Every method that
 * probes the model goes through predict(), which counts passes, so the pass
 * count N of Eq. (1) is read from the run rather than asserted.
 *
 * Randomness comes from a seeded xorshift32, never from a hardware TRNG: the
 * earlier measurements on the same nRF52840 found that TRNG waits can dominate a
 * sampling workload, which would put a term into T_expl that is not N*t_pass.
 */
#include <string.h>
#include <math.h>
#include "xai_core.h"
#include "model.h"

uint32_t xai_passes;
void (*xai_tick)(void);          /* called periodically on long runs */

static inline float leaf_sum(const float *x)
{
    float s = 0.0f;
    for (int t = 0; t < N_TREES; t++) {
        int k = tree_root[t];
        while (node_feat[k] >= 0)
            k = (x[node_feat[k]] < node_thr[k]) ? node_left[k] : node_right[k];
        s += node_val[k];
    }
    return s;
}

float predict(const float *x)
{
    xai_passes++;
    return BASE_MARGIN + leaf_sum(x);
}

/* ---------------- random numbers (seeded, reproducible) ---------------- */
static uint32_t rs = 20260823u;
void xai_seed(uint32_t s) { rs = s ? s : 1u; }
static inline uint32_t xr(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static inline float xu(void) { return (xr() >> 8) * (1.0f / 16777216.0f); }

/* ---------------- TreeSHAP (Lundberg et al., Algorithm 2) --------------- */
typedef struct { int f; float z, o, w; } pe_t;

static void extend(pe_t *p, int d, float z, float o, int f)
{
    p[d].f = f; p[d].z = z; p[d].o = o; p[d].w = d == 0 ? 1.0f : 0.0f;
    for (int i = d - 1; i >= 0; i--) {
        p[i + 1].w += o * p[i].w * (i + 1) / (float)(d + 1);
        p[i].w = z * p[i].w * (d - i) / (float)(d + 1);
    }
}

static void unwind(pe_t *p, int d, int pi)
{
    float o = p[pi].o, z = p[pi].z, n = p[d].w;
    for (int i = d - 1; i >= 0; i--) {
        if (o != 0.0f) {
            float t = p[i].w;
            p[i].w = n * (d + 1) / ((i + 1) * o);
            n = t - p[i].w * z * (d - i) / (float)(d + 1);
        } else {
            p[i].w = p[i].w * (d + 1) / (z * (d - i));
        }
    }
    for (int i = pi; i < d; i++) { p[i].f = p[i + 1].f; p[i].z = p[i + 1].z; p[i].o = p[i + 1].o; }
}

static float unwound_sum(const pe_t *p, int d, int pi)
{
    float o = p[pi].o, z = p[pi].z, n = p[d].w, tot = 0.0f;
    for (int i = d - 1; i >= 0; i--) {
        if (o != 0.0f) {
            float t = n * (d + 1) / ((i + 1) * o);
            tot += t;
            n = p[i].w - t * z * ((d - i) / (float)(d + 1));
        } else if (z != 0.0f) {
            tot += (p[i].w / z) / ((d - i) / (float)(d + 1));
        }
    }
    return tot;
}

static void recurse(int k, float *phi, const float *x, pe_t *pp, int d,
                    float pz, float po, int pf)
{
    pe_t *p = pp + d + 1;
    memcpy(p, pp, (size_t)(d + 1) * sizeof(pe_t));
    extend(p, d, pz, po, pf);
    int f = node_feat[k];
    if (f < 0) {
        for (int i = 1; i <= d; i++) {
            float w = unwound_sum(p, d, i);
            phi[p[i].f] += w * (p[i].o - p[i].z) * node_val[k];
        }
        return;
    }
    int hot = (x[f] < node_thr[k]) ? node_left[k] : node_right[k];
    int cold = hot == node_left[k] ? node_right[k] : node_left[k];
    float c = node_cover[k];
    float hz = node_cover[hot] / c, cz = node_cover[cold] / c, iz = 1.0f, io = 1.0f;
    int pi = 0;
    for (; pi <= d; pi++) if (p[pi].f == f) break;
    if (pi != d + 1) { iz = p[pi].z; io = p[pi].o; unwind(p, d, pi); d--; }
    recurse(hot, phi, x, p, d + 1, hz * iz, io, f);
    recurse(cold, phi, x, p, d + 1, cz * iz, 0.0f, f);
}

static float tree_mean(int k)
{
    if (node_feat[k] < 0) return node_val[k];
    return (node_cover[node_left[k]] * tree_mean(node_left[k]) +
            node_cover[node_right[k]] * tree_mean(node_right[k])) / node_cover[k];
}

float treeshap_expected(void)
{
    float e = BASE_MARGIN;
    for (int t = 0; t < N_TREES; t++) e += tree_mean(tree_root[t]);
    return e;
}

void treeshap(const float *x, float *phi)
{
    static pe_t buf[64];          /* (depth+2)(depth+3)/2 = 36 for depth 6 */
    memset(phi, 0, N_FEAT * sizeof(float));
    for (int t = 0; t < N_TREES; t++)
        recurse(tree_root[t], phi, x, buf, 0, 1.0f, 1.0f, -1);
}

/* ---------------- weighted least squares (normal equations) ------------- */
#define MAXP (N_FEAT + 1)
static float A[MAXP][MAXP], B[MAXP];

static void wls_reset(int m) { memset(A, 0, sizeof A); memset(B, 0, sizeof B); (void)m; }

static void wls_add(const float *r, int m, float y, float w)
{
    for (int i = 0; i < m; i++) {
        B[i] += w * r[i] * y;
        for (int j = 0; j <= i; j++) A[i][j] += w * r[i] * r[j];
    }
}

/* Cholesky solve with a small ridge; A symmetric, lower half filled. */
static void wls_solve(int m, float *out)
{
    static float L[MAXP][MAXP];
    for (int i = 0; i < m; i++) A[i][i] += 1e-4f;
    for (int i = 0; i < m; i++)
        for (int j = 0; j <= i; j++) {
            float s = A[i][j];
            for (int k = 0; k < j; k++) s -= L[i][k] * L[j][k];
            L[i][j] = (i == j) ? sqrtf(s > 1e-12f ? s : 1e-12f) : s / L[j][j];
        }
    float y[MAXP];
    for (int i = 0; i < m; i++) {
        float s = B[i];
        for (int k = 0; k < i; k++) s -= L[i][k] * y[k];
        y[i] = s / L[i][i];
    }
    for (int i = m - 1; i >= 0; i--) {
        float s = y[i];
        for (int k = i + 1; k < m; k++) s -= L[k][i] * out[k];
        out[i] = s / L[i][i];
    }
}

/* ---------------- KernelSHAP: K coalitions x N_BG background rows -------
 * Coalition sizes are drawn from the Shapley-kernel distribution, so every
 * sample carries unit weight; each coalition is paired with its complement.
 * The efficiency constraint sum(phi) = f(x) - E is enforced by eliminating
 * the last feature, as the `shap` implementation does.
 * Passes: 1 (f(x)) + K * N_BG.                                              */
float bg_expected;

void xai_init(void)
{
    float s = 0.0f;
    for (int b = 0; b < N_BG; b++) s += BASE_MARGIN + leaf_sum(bg_x[b]);
    bg_expected = s / N_BG;
}

void kernelshap(const float *x, int K, float *phi)
{
    static float size_cdf[N_FEAT];
    float tot = 0.0f;
    for (int s = 1; s < N_FEAT; s++) tot += (N_FEAT - 1.0f) / (s * (float)(N_FEAT - s));
    float acc = 0.0f;
    for (int s = 1; s < N_FEAT; s++) {
        acc += (N_FEAT - 1.0f) / (s * (float)(N_FEAT - s)) / tot;
        size_cdf[s] = acc;
    }
    float fx = predict(x), h[N_FEAT], r[N_FEAT];
    uint8_t z[N_FEAT];
    const int m = N_FEAT - 1;
    wls_reset(m);
    for (int c = 0; c < K; c++) {
        if ((c & 1) == 0) {                       /* new coalition */
            float u = xu();
            int size = 1;
            while (size < N_FEAT - 1 && size_cdf[size] < u) size++;
            memset(z, 0, sizeof z);
            for (int n = 0; n < size;) {          /* random subset of `size` */
                int j = (int)(xr() % N_FEAT);
                if (!z[j]) { z[j] = 1; n++; }
            }
        } else {                                  /* its complement */
            for (int j = 0; j < N_FEAT; j++) z[j] = !z[j];
        }
        float ey = 0.0f;
        for (int b = 0; b < N_BG; b++) {
            for (int j = 0; j < N_FEAT; j++) h[j] = z[j] ? x[j] : bg_x[b][j];
            ey += predict(h);
        }
        ey /= N_BG;
        float y = ey - bg_expected - z[m] * (fx - bg_expected);
        for (int j = 0; j < m; j++) r[j] = (float)z[j] - (float)z[m];
        wls_add(r, m, y, 1.0f);
        if (xai_tick) xai_tick();
    }
    wls_solve(m, phi);
    float s = 0.0f;
    for (int j = 0; j < m; j++) s += phi[j];
    phi[m] = (fx - bg_expected) - s;
}

/* ---------------- LIME-style local surrogate: N perturbations -----------
 * Each perturbation keeps a feature of x with probability 1/2 and otherwise
 * takes it from a random background row (sampling from the data, as LIME's
 * tabular sampler does); an exponential kernel on the fraction of features
 * changed weights the sample. Passes: N.                                   */
void lime(const float *x, int N, float *coef)
{
    float h[N_FEAT], r[N_FEAT + 1];
    wls_reset(N_FEAT + 1);
    for (int n = 0; n < N; n++) {
        int b = (int)(xr() % N_BG), changed = 0;
        for (int j = 0; j < N_FEAT; j++) {
            int keep = (xr() >> 31) & 1;
            h[j] = keep ? x[j] : bg_x[b][j];
            r[j] = (float)keep;
            changed += !keep;
        }
        r[N_FEAT] = 1.0f;
        float d = (float)changed / N_FEAT;
        float w = expf(-(d * d) / (0.75f * 0.75f));
        wls_add(r, N_FEAT + 1, predict(h), w);
        if (xai_tick && (n & 255) == 0) xai_tick();
    }
    wls_solve(N_FEAT + 1, coef);
}

/* ---------------- Occlusion: one pass per feature, plus f(x) ------------ */
void occlusion(const float *x, float *phi)
{
    static float mean[N_FEAT];
    static int ready;
    if (!ready) {
        for (int j = 0; j < N_FEAT; j++) {
            float s = 0.0f;
            for (int b = 0; b < N_BG; b++) s += bg_x[b][j];
            mean[j] = s / N_BG;
        }
        ready = 1;
    }
    float fx = predict(x), h[N_FEAT];
    memcpy(h, x, sizeof h);
    for (int j = 0; j < N_FEAT; j++) {
        h[j] = mean[j];
        phi[j] = fx - predict(h);
        h[j] = x[j];
    }
}
