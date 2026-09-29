/* hq.c — "Pixar-style" CPU path tracer for the Body Explorers episodes.
 *
 * The scene that render.py describes is, geometrically, a pile of affine
 * transformed unit spheres plus one biconcave red-blood-cell mesh. That makes
 * ray tracing far more attractive than rasterizing: ellipsoids are intersected
 * analytically, so silhouettes are perfectly smooth at any resolution, and
 * soft shadows / ambient occlusion / depth of field / indirect bounce light all
 * fall out of the same ray budget.
 *
 * Feature set
 *   - analytic ellipsoid instances + instanced triangle mesh (BVH), instance BVH
 *   - sphere area lights with area sampling  -> soft, penumbra-correct shadows
 *   - gradient sky environment               -> image-based ambient + AO
 *   - one diffuse GI bounce                  -> colour bleeding, soft fill
 *   - GGX specular with Schlick fresnel, metallic + roughness workflow
 *   - wrap-lighting subsurface approximation for skin / cells / organs
 *   - thin-lens depth of field, stochastic AA, stratified sampling
 *   - edge-aware a-trous denoiser guided by albedo + normal + depth
 *
 * Output is linear HDR RGB; tonemapping and bloom happen on the Python side.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define MAX_LIGHTS 8
#define EPS 1e-4f

/* ------------------------------------------------------------------ maths */
typedef struct { float x, y, z; } V3;

static inline V3 v3(float x, float y, float z) { V3 r = {x, y, z}; return r; }
static inline V3 add(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline V3 sub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline V3 mul(V3 a, V3 b) { return v3(a.x * b.x, a.y * b.y, a.z * b.z); }
static inline V3 scale(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross(V3 a, V3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float len(V3 a) { return sqrtf(dot(a, a)); }
static inline V3 norm(V3 a) {
    float l = len(a);
    return l > 1e-20f ? scale(a, 1.f / l) : v3(0, 1, 0);
}
static inline float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
static inline float maxf(float a, float b) { return a > b ? a : b; }
static inline float minf(float a, float b) { return a < b ? a : b; }

/* row-major 4x4 times point / direction */
static inline V3 xform_p(const float *m, V3 p) {
    return v3(m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
              m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
              m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]);
}
static inline V3 xform_d(const float *m, V3 p) {
    return v3(m[0] * p.x + m[1] * p.y + m[2] * p.z,
              m[4] * p.x + m[5] * p.y + m[6] * p.z,
              m[8] * p.x + m[9] * p.y + m[10] * p.z);
}
/* transpose(upper3x3) times vector — used to carry normals through inverses */
static inline V3 xform_nt(const float *m, V3 p) {
    return v3(m[0] * p.x + m[4] * p.y + m[8] * p.z,
              m[1] * p.x + m[5] * p.y + m[9] * p.z,
              m[2] * p.x + m[6] * p.y + m[10] * p.z);
}

/* -------------------------------------------------------------------- rng */
typedef struct { uint64_t state, inc; } Rng;

static inline uint32_t pcg(Rng *r) {
    uint64_t old = r->state;
    r->state = old * 6364136223846793005ULL + r->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-(int32_t)rot) & 31));
}
static inline float rnd(Rng *r) { return (pcg(r) >> 8) * (1.f / 16777216.f); }

static inline void rng_seed(Rng *r, uint64_t a, uint64_t b) {
    r->state = 0u; r->inc = (b << 1u) | 1u;
    pcg(r); r->state += a; pcg(r);
}

/* ------------------------------------------------------------------ scene */
typedef struct {
    float m[16], inv[16];
    float albedo[3];
    float rough, metal, emissive, sss;
    int mesh;                 /* 0 = unit sphere, 1 = triangle mesh */
    float bmin[3], bmax[3];
} Inst;

typedef struct { float bmin[3], bmax[3]; int left, start, count; } Node;

typedef struct {
    float pos[3], radius, color[3], intensity;
} Light;

static Inst *g_inst = NULL;
static int g_ninst = 0;
static Node *g_inode = NULL;
static int *g_iidx = NULL;

/* shared triangle mesh (the red blood cell) */
static float *g_mv = NULL, *g_mn = NULL;
static int *g_mi = NULL;
static int g_ntri = 0;
static Node *g_mnode = NULL;
static int *g_midx = NULL;

static Light g_lights[MAX_LIGHTS];
static int g_nlights = 0;
static V3 g_sky_top, g_sky_hor, g_sky_bot;
static float g_sky_gain = 1.f;

/* --------------------------------------------------------------- bvh core */
typedef struct {
    float *bmin, *bmax, *cent;   /* per primitive */
    int *idx;
    Node *nodes;
    int nnodes;
} BvhBuild;

static int build_rec(BvhBuild *b, int start, int count, int depth) {
    int me = b->nnodes++;
    Node *n = &b->nodes[me];
    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = start; i < start + count; i++) {
        int p = b->idx[i];
        for (int k = 0; k < 3; k++) {
            if (b->bmin[p * 3 + k] < mn[k]) mn[k] = b->bmin[p * 3 + k];
            if (b->bmax[p * 3 + k] > mx[k]) mx[k] = b->bmax[p * 3 + k];
        }
    }
    memcpy(n->bmin, mn, sizeof mn);
    memcpy(n->bmax, mx, sizeof mx);
    n->start = start; n->count = count; n->left = -1;
    if (count <= 4 || depth > 40) return me;

    int axis = 0;
    float ext[3] = {mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]};
    if (ext[1] > ext[axis]) axis = 1;
    if (ext[2] > ext[axis]) axis = 2;
    float split = 0.5f * (mn[axis] + mx[axis]);

    int mid = start, end = start + count;
    for (int i = start; i < end; i++) {
        if (b->cent[b->idx[i] * 3 + axis] < split) {
            int t = b->idx[i]; b->idx[i] = b->idx[mid]; b->idx[mid] = t; mid++;
        }
    }
    if (mid == start || mid == end) mid = start + count / 2;

    n->count = 0;
    int l = build_rec(b, start, mid - start, depth + 1);
    int r = build_rec(b, mid, start + count - mid, depth + 1);
    b->nodes[me].left = l;
    b->nodes[me].start = r;   /* right child index stored in start for inner nodes */
    (void)l; (void)r;
    return me;
}

static Node *build_bvh(float *bmin, float *bmax, int n, int **out_idx) {
    Node *nodes = (Node *)malloc(sizeof(Node) * (size_t)(2 * n + 8));
    int *idx = (int *)malloc(sizeof(int) * (size_t)n);
    float *cent = (float *)malloc(sizeof(float) * (size_t)n * 3);
    for (int i = 0; i < n; i++) {
        idx[i] = i;
        for (int k = 0; k < 3; k++) cent[i * 3 + k] = 0.5f * (bmin[i * 3 + k] + bmax[i * 3 + k]);
    }
    BvhBuild b = {bmin, bmax, cent, idx, nodes, 0};
    build_rec(&b, 0, n, 0);
    free(cent);
    *out_idx = idx;
    return nodes;
}

static inline int slab(const float *bmin, const float *bmax, V3 o, V3 invd, float tmax, float *tnear) {
    float t0 = 0.f, t1 = tmax;
    const float od[3] = {o.x, o.y, o.z}, id[3] = {invd.x, invd.y, invd.z};
    for (int k = 0; k < 3; k++) {
        float a = (bmin[k] - od[k]) * id[k];
        float b = (bmax[k] - od[k]) * id[k];
        if (a > b) { float t = a; a = b; b = t; }
        if (a > t0) t0 = a;
        if (b < t1) t1 = b;
        if (t0 > t1) return 0;
    }
    *tnear = t0;
    return 1;
}

/* ------------------------------------------------------------ intersection */
typedef struct { float t; V3 n; int inst; } Hit;

static inline int tri_hit(int t, V3 o, V3 d, float *tout, float *bu, float *bv) {
    const int *f = &g_mi[t * 3];
    const float *p0 = &g_mv[f[0] * 3], *p1 = &g_mv[f[1] * 3], *p2 = &g_mv[f[2] * 3];
    V3 a = v3(p0[0], p0[1], p0[2]);
    V3 e1 = v3(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
    V3 e2 = v3(p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]);
    V3 pv = cross(d, e2);
    float det = dot(e1, pv);
    if (fabsf(det) < 1e-12f) return 0;
    float invdet = 1.f / det;
    V3 tv = sub(o, a);
    float u = dot(tv, pv) * invdet;
    if (u < -1e-6f || u > 1.f + 1e-6f) return 0;
    V3 qv = cross(tv, e1);
    float v = dot(d, qv) * invdet;
    if (v < -1e-6f || u + v > 1.f + 1e-6f) return 0;
    float tt = dot(e2, qv) * invdet;
    if (tt <= EPS) return 0;
    *tout = tt; *bu = u; *bv = v;
    return 1;
}

/* mesh traversal in object space */
static int mesh_hit(V3 o, V3 d, float tmax, float *tbest, V3 *nbest, int any) {
    if (!g_ntri) return 0;
    V3 invd = v3(1.f / (d.x ? d.x : 1e-20f), 1.f / (d.y ? d.y : 1e-20f), 1.f / (d.z ? d.z : 1e-20f));
    int stack[64], sp = 0, found = 0;
    stack[sp++] = 0;
    float best = tmax;
    while (sp) {
        int ni = stack[--sp];
        Node *n = &g_mnode[ni];
        float tn;
        if (!slab(n->bmin, n->bmax, o, invd, best, &tn)) continue;
        if (n->count) {
            for (int i = n->start; i < n->start + n->count; i++) {
                int t = g_midx[i];
                float tt, bu, bv;
                if (tri_hit(t, o, d, &tt, &bu, &bv) && tt < best) {
                    best = tt; found = 1;
                    if (any) { *tbest = tt; return 1; }
                    const int *f = &g_mi[t * 3];
                    const float *n0 = &g_mn[f[0] * 3], *n1 = &g_mn[f[1] * 3], *n2 = &g_mn[f[2] * 3];
                    float w = 1.f - bu - bv;
                    *nbest = v3(w * n0[0] + bu * n1[0] + bv * n2[0],
                                w * n0[1] + bu * n1[1] + bv * n2[1],
                                w * n0[2] + bu * n1[2] + bv * n2[2]);
                }
            }
        } else {
            stack[sp++] = n->left;
            stack[sp++] = n->start;
        }
    }
    if (found) *tbest = best;
    return found;
}

static int inst_hit(int ii, V3 o, V3 d, float tmax, float *tout, V3 *nout, int any) {
    const Inst *in = &g_inst[ii];
    V3 lo = xform_p(in->inv, o);
    V3 ld = xform_d(in->inv, d);
    if (in->mesh == 0) {
        float a = dot(ld, ld);
        float b = 2.f * dot(lo, ld);
        float c = dot(lo, lo) - 1.f;
        float disc = b * b - 4.f * a * c;
        if (disc < 0.f) return 0;
        float sq = sqrtf(disc);
        float t = (-b - sq) / (2.f * a);
        if (t <= EPS) t = (-b + sq) / (2.f * a);
        if (t <= EPS || t >= tmax) return 0;
        *tout = t;
        if (!any) {
            V3 lp = add(lo, scale(ld, t));
            *nout = norm(xform_nt(in->inv, lp));
        }
        return 1;
    }
    float t; V3 n;
    if (!mesh_hit(lo, ld, tmax, &t, &n, any)) return 0;
    if (t <= EPS) return 0;
    *tout = t;
    if (!any) *nout = norm(xform_nt(in->inv, n));
    return 1;
}

static int trace(V3 o, V3 d, float tmax, Hit *hit, int any) {
    if (!g_ninst) return 0;
    V3 invd = v3(1.f / (d.x ? d.x : 1e-20f), 1.f / (d.y ? d.y : 1e-20f), 1.f / (d.z ? d.z : 1e-20f));
    int stack[64], sp = 0, found = 0;
    stack[sp++] = 0;
    float best = tmax;
    while (sp) {
        int ni = stack[--sp];
        Node *n = &g_inode[ni];
        float tn;
        if (!slab(n->bmin, n->bmax, o, invd, best, &tn)) continue;
        if (n->count) {
            for (int i = n->start; i < n->start + n->count; i++) {
                int ii = g_iidx[i];
                float t; V3 nn = v3(0, 1, 0);
                if (inst_hit(ii, o, d, best, &t, &nn, any)) {
                    if (any) { hit->t = t; hit->inst = ii; return 1; }
                    best = t; found = 1; hit->t = t; hit->n = nn; hit->inst = ii;
                }
            }
        } else {
            stack[sp++] = n->left;
            stack[sp++] = n->start;
        }
    }
    return found;
}

/* ----------------------------------------------------------------- shading */
static V3 sky(V3 d) {
    float t = clampf(d.y * 0.5f + 0.5f, 0.f, 1.f);
    V3 c;
    if (t > 0.5f) {
        float k = (t - 0.5f) * 2.f;
        c = add(scale(g_sky_hor, 1.f - k), scale(g_sky_top, k));
    } else {
        float k = t * 2.f;
        c = add(scale(g_sky_bot, 1.f - k), scale(g_sky_hor, k));
    }
    /* a soft warm glow around the key light keeps highlights lively */
    if (g_nlights) {
        V3 lp = v3(g_lights[0].pos[0], g_lights[0].pos[1], g_lights[0].pos[2]);
        float g = powf(maxf(dot(d, norm(lp)), 0.f), 24.f) * 0.35f;
        c = add(c, scale(v3(1.f, 0.86f, 0.62f), g));
    }
    return scale(c, g_sky_gain);
}

static inline void onb(V3 n, V3 *t, V3 *b) {
    V3 up = fabsf(n.y) < 0.95f ? v3(0, 1, 0) : v3(1, 0, 0);
    *t = norm(cross(up, n));
    *b = cross(n, *t);
}

static inline V3 cos_hemi(V3 n, Rng *r) {
    float u1 = rnd(r), u2 = rnd(r);
    float rr = sqrtf(u1), phi = 6.2831853f * u2;
    V3 t, b; onb(n, &t, &b);
    return norm(add(add(scale(t, rr * cosf(phi)), scale(b, rr * sinf(phi))),
                    scale(n, sqrtf(maxf(0.f, 1.f - u1)))));
}

static float ggx(V3 n, V3 v, V3 l, float rough) {
    V3 h = norm(add(v, l));
    float a = maxf(rough * rough, 1e-3f);
    float a2 = a * a;
    float ndh = maxf(dot(n, h), 0.f);
    float ndv = maxf(dot(n, v), 1e-4f);
    float ndl = maxf(dot(n, l), 0.f);
    float dd = ndh * ndh * (a2 - 1.f) + 1.f;
    float D = a2 / (3.14159265f * dd * dd);
    float k = a * 0.5f;
    float gv = ndv / (ndv * (1.f - k) + k);
    float gl = ndl / (ndl * (1.f - k) + k);
    return D * gv * gl / (4.f * ndv * maxf(ndl, 1e-4f));
}

static inline V3 fresnel(V3 f0, float ct) {
    float f = powf(1.f - clampf(ct, 0.f, 1.f), 5.f);
    return add(f0, scale(sub(v3(1, 1, 1), f0), f));
}

/* direct lighting from all sphere area lights */
static V3 direct(V3 p, V3 n, V3 v, const Inst *m, Rng *r, int cheap) {
    V3 out = v3(0, 0, 0);
    V3 alb = v3(m->albedo[0], m->albedo[1], m->albedo[2]);
    V3 f0 = add(scale(v3(0.04f, 0.04f, 0.04f), 1.f - m->metal), scale(alb, m->metal));
    V3 diff_alb = scale(alb, 1.f - m->metal);
    int nl = cheap ? (g_nlights > 0 ? 1 : 0) : g_nlights;

    for (int i = 0; i < nl; i++) {
        const Light *L = &g_lights[i];
        V3 lc = v3(L->pos[0], L->pos[1], L->pos[2]);
        V3 toL = sub(lc, p);
        float dist = len(toL);
        if (dist < 1e-4f) continue;
        V3 cdir = scale(toL, 1.f / dist);

        /* Solid-angle (cone) sampling of the sphere light. Sampling the light's
         * *area* makes big soft sources very noisy; sampling the cone they
         * subtend keeps penumbrae soft while cutting variance dramatically. */
        float sin_max = minf(L->radius / dist, 0.9999f);
        float cos_max = sqrtf(maxf(0.f, 1.f - sin_max * sin_max));
        float u1 = rnd(r), u2 = rnd(r);
        float cos_t = 1.f - u1 * (1.f - cos_max);
        float sin_t = sqrtf(maxf(0.f, 1.f - cos_t * cos_t));
        float phi = 6.2831853f * u2;
        V3 tb, bb; onb(cdir, &tb, &bb);
        V3 ldir = norm(add(add(scale(tb, cosf(phi) * sin_t), scale(bb, sinf(phi) * sin_t)),
                           scale(cdir, cos_t)));

        float ndl = dot(n, ldir);
        float wrap = m->sss;
        float diff_term = wrap > 0.f ? maxf((ndl + wrap) / (1.f + wrap), 0.f) : maxf(ndl, 0.f);
        if (diff_term <= 0.f && ndl <= 0.f) continue;

        Hit sh;
        if (trace(add(p, scale(n, 1e-3f)), ldir, dist - L->radius * 0.5f, &sh, 1)) continue;

        /* 1/pdf for uniform cone sampling */
        float g = 6.2831853f * (1.f - cos_max);
        V3 lcol = scale(v3(L->color[0], L->color[1], L->color[2]), L->intensity * g);
        V3 kd = scale(diff_alb, diff_term / 3.14159265f);
        out = add(out, mul(kd, lcol));

        if (ndl > 0.f && !cheap) {
            /* Representative-point specular: deterministic direction to the
             * light centre, roughness widened by its angular radius. Keeps
             * glossy highlights soft and, crucially, noise free. */
            float ang = atanf(L->radius / maxf(dist, 1e-3f));
            float rr = clampf(m->rough + ang * 0.55f, 0.03f, 1.f);
            float ndl_c = maxf(dot(n, cdir), 0.f);
            float s = ggx(n, v, cdir, rr);
            V3 F = fresnel(f0, maxf(dot(norm(add(v, cdir)), v), 0.f));
            out = add(out, mul(scale(F, s * ndl_c), lcol));
        }
    }
    return out;
}

typedef struct {
    int W, H, spp, bounces;
    float eye[3], fwd[3], right[3], up[3];
    float tan_half, aspect, aperture, focus;
    float *out, *albedo_buf, *normal_buf, *depth_buf;
    int y0, y1;
    uint64_t seed;
} Job;

static V3 shade(V3 o, V3 d, Rng *rng, int bounces, V3 *first_albedo, V3 *first_normal, float *first_depth);

static V3 shade(V3 o, V3 d, Rng *rng, int bounces, V3 *first_albedo, V3 *first_normal, float *first_depth) {
    Hit h;
    if (!trace(o, d, 1e30f, &h, 0)) {
        if (first_albedo) { *first_albedo = v3(1, 1, 1); *first_normal = scale(d, -1.f); *first_depth = 1e30f; }
        return sky(d);
    }
    const Inst *m = &g_inst[h.inst];
    V3 p = add(o, scale(d, h.t));
    V3 n = h.n;
    V3 v = scale(d, -1.f);
    if (dot(n, v) < 0.f && m->sss > 0.f) n = scale(n, -1.f);   /* two-sided for soft bodies */

    V3 alb = v3(m->albedo[0], m->albedo[1], m->albedo[2]);
    if (first_albedo) {
        *first_albedo = m->emissive > 0.f ? v3(1, 1, 1) : alb;
        *first_normal = n;
        *first_depth = h.t;
    }

    V3 col = scale(alb, m->emissive * 2.0f);            /* glowing molecules */
    col = add(col, direct(p, n, v, m, rng, bounces == 0));

    /* one diffuse GI bounce: sky light, ambient occlusion and colour bleed */
    if (bounces > 0) {
        V3 dir = cos_hemi(n, rng);
        Hit gh;
        V3 gi;
        if (!trace(add(p, scale(n, 1e-3f)), dir, 1e30f, &gh, 0)) {
            gi = sky(dir);
        } else {
            const Inst *gm = &g_inst[gh.inst];
            V3 gp = add(p, scale(dir, gh.t));
            V3 gn = gh.n;
            gi = scale(v3(gm->albedo[0], gm->albedo[1], gm->albedo[2]), gm->emissive * 2.0f);
            gi = add(gi, direct(gp, gn, scale(dir, -1.f), gm, rng, 1));
        }
        V3 kd = scale(alb, 1.f - m->metal);
        const float fire = 6.0f;                    /* firefly clamp */
        gi = v3(minf(gi.x, fire), minf(gi.y, fire), minf(gi.z, fire));
        col = add(col, mul(kd, gi));
    }

    /* cheap glossy environment reflection keeps surfaces looking "rendered" */
    if (m->rough < 0.55f) {
        V3 rdir = sub(scale(n, 2.f * dot(n, v)), v);
        V3 env = sky(rdir);
        V3 f0 = add(scale(v3(0.04f, 0.04f, 0.04f), 1.f - m->metal), scale(alb, m->metal));
        V3 F = fresnel(f0, maxf(dot(n, v), 0.f));
        float gloss = 1.f - m->rough / 0.55f;
        col = add(col, scale(mul(F, env), gloss * 0.8f));
    }
    return col;
}

static void *render_rows(void *arg) {
    Job *j = (Job *)arg;
    V3 eye = v3(j->eye[0], j->eye[1], j->eye[2]);
    V3 fwd = v3(j->fwd[0], j->fwd[1], j->fwd[2]);
    V3 rgt = v3(j->right[0], j->right[1], j->right[2]);
    V3 up = v3(j->up[0], j->up[1], j->up[2]);
    int W = j->W, H = j->H, spp = j->spp;
    int sq = (int)sqrtf((float)spp);
    if (sq < 1) sq = 1;

    for (int y = j->y0; y < j->y1; y++) {
        for (int x = 0; x < W; x++) {
            /* Seeding only from the pixel keeps residual noise stationary from
             * frame to frame, which reads as fine grain rather than boiling. */
            Rng rng;
            rng_seed(&rng, (uint64_t)(y * (int64_t)W + x) * 9781ULL + j->seed, 0xda3e39cb94b95bdbULL);

            V3 acc = v3(0, 0, 0), alb_acc = v3(0, 0, 0), nrm_acc = v3(0, 0, 0);
            float depth_acc = 0.f;
            float m2 = 0.f;                 /* running sum of squared luma */
            int s = 0;
            const int min_s = spp > 4 ? 4 : spp;
            for (int sy = 0; sy < sq; sy++) {
                for (int sx = 0; sx < sq && s < spp; sx++, s++) {
                    /* Large parts of every shot are empty backdrop; once the
                     * estimate is stable there is nothing left to gain. */
                    if (s >= min_s) {
                        float mean = (acc.x + acc.y + acc.z) / (3.f * (float)s);
                        float var = m2 / (float)s - mean * mean;
                        if (var < 2.5e-4f * (1.f + mean * mean)) break;
                    }
                    float jx = ((float)sx + rnd(&rng)) / (float)sq;
                    float jy = ((float)sy + rnd(&rng)) / (float)sq;
                    float px = (2.f * ((float)x + jx) / (float)W - 1.f) * j->tan_half * j->aspect;
                    float py = (1.f - 2.f * ((float)y + jy) / (float)H) * j->tan_half;
                    V3 dir = norm(add(fwd, add(scale(rgt, px), scale(up, py))));

                    V3 o = eye;
                    if (j->aperture > 0.f) {
                        /* thin lens: shallow, cinematic focus falloff */
                        float a = 6.2831853f * rnd(&rng), rr = sqrtf(rnd(&rng)) * j->aperture;
                        V3 off = add(scale(rgt, cosf(a) * rr), scale(up, sinf(a) * rr));
                        V3 fp = add(eye, scale(dir, j->focus / maxf(dot(dir, fwd), 1e-3f)));
                        o = add(eye, off);
                        dir = norm(sub(fp, o));
                    }
                    V3 a_out = v3(1, 1, 1), n_out = v3(0, 0, 1);
                    float d_out = 1e30f;
                    V3 rad = shade(o, dir, &rng, j->bounces, &a_out, &n_out, &d_out);
                    acc = add(acc, rad);
                    float lum = (rad.x + rad.y + rad.z) / 3.f;
                    m2 += lum * lum;
                    alb_acc = add(alb_acc, a_out);
                    nrm_acc = add(nrm_acc, n_out);
                    depth_acc += minf(d_out, 1e4f);
                }
            }
            float inv = 1.f / (float)s;
            long o = ((long)y * W + x);
            j->out[o * 3 + 0] = acc.x * inv;
            j->out[o * 3 + 1] = acc.y * inv;
            j->out[o * 3 + 2] = acc.z * inv;
            j->albedo_buf[o * 3 + 0] = alb_acc.x * inv;
            j->albedo_buf[o * 3 + 1] = alb_acc.y * inv;
            j->albedo_buf[o * 3 + 2] = alb_acc.z * inv;
            j->normal_buf[o * 3 + 0] = nrm_acc.x * inv;
            j->normal_buf[o * 3 + 1] = nrm_acc.y * inv;
            j->normal_buf[o * 3 + 2] = nrm_acc.z * inv;
            j->depth_buf[o] = depth_acc * inv;
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------- API */
void hq_set_mesh(float *verts, float *normals, int nverts, int *idx, int ntri) {
    if (g_mnode) { free(g_mnode); g_mnode = NULL; }
    if (g_midx) { free(g_midx); g_midx = NULL; }
    g_mv = verts; g_mn = normals; g_mi = idx; g_ntri = ntri;
    (void)nverts;
    if (!ntri) return;
    float *bmin = (float *)malloc(sizeof(float) * (size_t)ntri * 3);
    float *bmax = (float *)malloc(sizeof(float) * (size_t)ntri * 3);
    for (int t = 0; t < ntri; t++) {
        for (int k = 0; k < 3; k++) {
            float a = verts[idx[t * 3 + 0] * 3 + k];
            float b = verts[idx[t * 3 + 1] * 3 + k];
            float c = verts[idx[t * 3 + 2] * 3 + k];
            bmin[t * 3 + k] = minf(a, minf(b, c)) - 1e-5f;
            bmax[t * 3 + k] = maxf(a, maxf(b, c)) + 1e-5f;
        }
    }
    g_mnode = build_bvh(bmin, bmax, ntri, &g_midx);
    free(bmin); free(bmax);
}

/* inst_data: per instance 16 (matrix) + 16 (inverse) + 3 albedo + 4 params = 39 floats
 * mesh_ids : per instance int */
void hq_set_instances(float *inst_data, int *mesh_ids, int n) {
    if (g_inst) { free(g_inst); g_inst = NULL; }
    if (g_inode) { free(g_inode); g_inode = NULL; }
    if (g_iidx) { free(g_iidx); g_iidx = NULL; }
    g_ninst = n;
    if (!n) return;
    g_inst = (Inst *)malloc(sizeof(Inst) * (size_t)n);
    float *bmin = (float *)malloc(sizeof(float) * (size_t)n * 3);
    float *bmax = (float *)malloc(sizeof(float) * (size_t)n * 3);

    for (int i = 0; i < n; i++) {
        float *d = inst_data + (size_t)i * 39;
        Inst *in = &g_inst[i];
        memcpy(in->m, d, 16 * sizeof(float));
        memcpy(in->inv, d + 16, 16 * sizeof(float));
        in->albedo[0] = d[32]; in->albedo[1] = d[33]; in->albedo[2] = d[34];
        in->rough = d[35]; in->metal = d[36]; in->emissive = d[37]; in->sss = d[38];
        in->mesh = mesh_ids[i];

        /* world AABB of the transformed unit cube bounding the primitive */
        float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
        for (int c = 0; c < 8; c++) {
            V3 corner = v3((c & 1) ? 1.f : -1.f, (c & 2) ? 1.f : -1.f, (c & 4) ? 1.f : -1.f);
            V3 w = xform_p(in->m, corner);
            float wa[3] = {w.x, w.y, w.z};
            for (int k = 0; k < 3; k++) {
                if (wa[k] < mn[k]) mn[k] = wa[k];
                if (wa[k] > mx[k]) mx[k] = wa[k];
            }
        }
        for (int k = 0; k < 3; k++) {
            in->bmin[k] = mn[k] - 1e-4f;
            in->bmax[k] = mx[k] + 1e-4f;
            bmin[i * 3 + k] = in->bmin[k];
            bmax[i * 3 + k] = in->bmax[k];
        }
    }
    g_inode = build_bvh(bmin, bmax, n, &g_iidx);
    free(bmin); free(bmax);
}

void hq_set_lights(float *data, int n) {
    if (n > MAX_LIGHTS) n = MAX_LIGHTS;
    g_nlights = n;
    for (int i = 0; i < n; i++) {
        float *d = data + i * 8;
        memcpy(g_lights[i].pos, d, 3 * sizeof(float));
        g_lights[i].radius = d[3];
        memcpy(g_lights[i].color, d + 4, 3 * sizeof(float));
        g_lights[i].intensity = d[7];
    }
}

void hq_set_sky(float *top, float *hor, float *bot, float gain) {
    g_sky_top = v3(top[0], top[1], top[2]);
    g_sky_hor = v3(hor[0], hor[1], hor[2]);
    g_sky_bot = v3(bot[0], bot[1], bot[2]);
    g_sky_gain = gain;
}

void hq_render(int W, int H, int spp, int bounces, float *cam, float aperture, float focus,
               int nthreads, uint64_t seed,
               float *out, float *albedo_buf, float *normal_buf, float *depth_buf) {
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 32) nthreads = 32;
    pthread_t th[32];
    Job jobs[32];
    int rows = (H + nthreads - 1) / nthreads;
    for (int i = 0; i < nthreads; i++) {
        Job *j = &jobs[i];
        j->W = W; j->H = H; j->spp = spp; j->bounces = bounces;
        memcpy(j->eye, cam + 0, 3 * sizeof(float));
        memcpy(j->fwd, cam + 3, 3 * sizeof(float));
        memcpy(j->right, cam + 6, 3 * sizeof(float));
        memcpy(j->up, cam + 9, 3 * sizeof(float));
        j->tan_half = cam[12];
        j->aspect = cam[13];
        j->aperture = aperture;
        j->focus = focus;
        j->out = out; j->albedo_buf = albedo_buf; j->normal_buf = normal_buf; j->depth_buf = depth_buf;
        j->y0 = i * rows;
        j->y1 = (i + 1) * rows;
        if (j->y1 > H) j->y1 = H;
        j->seed = seed;
        if (j->y0 >= j->y1) { j->y0 = j->y1 = 0; }
        pthread_create(&th[i], NULL, render_rows, j);
    }
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
}

/* Edge-aware a-trous wavelet denoiser guided by albedo, normal and depth.
 * Colour is demodulated by albedo first so texture detail survives. */
typedef struct {
    float *cur, *tmp;
    const float *normal, *depth;
    int W, H, step, y0, y1;
    float sigma_n, sigma_z, sigma_c;
} DnJob;

static void *denoise_rows(void *arg) {
    DnJob *j = (DnJob *)arg;
    const float kern[3] = {3.f / 8.f, 1.f / 4.f, 1.f / 16.f};
    int W = j->W, H = j->H, step = j->step;
    for (int y = j->y0; y < j->y1; y++) {
        for (int x = 0; x < W; x++) {
            long o = (long)y * W + x;
            float cw = 0.f, sum[3] = {0, 0, 0};
            V3 n0 = v3(j->normal[o * 3], j->normal[o * 3 + 1], j->normal[o * 3 + 2]);
            float z0 = j->depth[o];
            for (int dy = -2; dy <= 2; dy++) {
                int yy = y + dy * step;
                if (yy < 0 || yy >= H) continue;
                for (int dx = -2; dx <= 2; dx++) {
                    int xx = x + dx * step;
                    if (xx < 0 || xx >= W) continue;
                    long oo = (long)yy * W + xx;
                    V3 n1 = v3(j->normal[oo * 3], j->normal[oo * 3 + 1], j->normal[oo * 3 + 2]);
                    float nd = maxf(dot(n0, n1), 0.f);
                    float wn = powf(nd, j->sigma_n);
                    float dz = fabsf(z0 - j->depth[oo]) / j->sigma_z;
                    float wz = expf(-dz * dz);
                    float dc = 0.f;
                    for (int k = 0; k < 3; k++) {
                        float d = j->cur[o * 3 + k] - j->cur[oo * 3 + k];
                        dc += d * d;
                    }
                    float wc = expf(-dc / maxf(j->sigma_c, 1e-4f));
                    float w = kern[abs(dx)] * kern[abs(dy)] * wn * wz * wc;
                    cw += w;
                    for (int k = 0; k < 3; k++) sum[k] += j->cur[oo * 3 + k] * w;
                }
            }
            if (cw > 0.f)
                for (int k = 0; k < 3; k++) j->tmp[o * 3 + k] = sum[k] / cw;
            else
                for (int k = 0; k < 3; k++) j->tmp[o * 3 + k] = j->cur[o * 3 + k];
        }
    }
    return NULL;
}

/* Edge-aware a-trous wavelet denoiser guided by albedo, normal and depth.
 * Colour is demodulated by albedo first so texture detail survives. */
void hq_denoise_mt(float *color, const float *albedo, const float *normal, const float *depth,
                   int W, int H, int iters, float sigma_n, float sigma_z, float sigma_c,
                   int nthreads) {
    long np = (long)W * H;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 32) nthreads = 32;
    float *tmp = (float *)malloc(sizeof(float) * (size_t)np * 3);
    float *cur = (float *)malloc(sizeof(float) * (size_t)np * 3);

    for (long i = 0; i < np; i++)
        for (int k = 0; k < 3; k++)
            cur[i * 3 + k] = color[i * 3 + k] / maxf(albedo[i * 3 + k], 0.08f);

    pthread_t th[32];
    DnJob jobs[32];
    int rows = (H + nthreads - 1) / nthreads;
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < nthreads; i++) {
            DnJob *j = &jobs[i];
            j->cur = cur; j->tmp = tmp; j->normal = normal; j->depth = depth;
            j->W = W; j->H = H; j->step = 1 << it;
            j->y0 = i * rows; j->y1 = (i + 1) * rows > H ? H : (i + 1) * rows;
            if (j->y0 > j->y1) j->y0 = j->y1;
            j->sigma_n = sigma_n; j->sigma_z = sigma_z; j->sigma_c = sigma_c;
            pthread_create(&th[i], NULL, denoise_rows, j);
        }
        for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
        memcpy(cur, tmp, sizeof(float) * (size_t)np * 3);
    }
    for (long i = 0; i < np; i++)
        for (int k = 0; k < 3; k++)
            color[i * 3 + k] = cur[i * 3 + k] * maxf(albedo[i * 3 + k], 0.08f);
    free(tmp); free(cur);
}

void hq_denoise(float *color, const float *albedo, const float *normal, const float *depth,
                int W, int H, int iters, float sigma_n, float sigma_z, float sigma_c) {
    hq_denoise_mt(color, albedo, normal, depth, W, H, iters, sigma_n, sigma_z, sigma_c, 1);
}
