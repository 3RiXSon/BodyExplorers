/* softraster.c — minimal software rasterizer implementing the exact shading
 * model used by Body-Explorers-Source-Package/render.py.
 *
 * Mirrors the GLSL program in render.py:
 *   vertex   : P = model*pos ; N = normalMatrix*normal ; gl_Position = vp*P
 *   fragment : l = normalize(vec3(-4,7,8)-P) ; v = normalize(eye-P)
 *              d = max(dot(n,l),0) ; rim = pow(1-max(dot(n,v),0),3)
 *              spec = pow(max(dot(n, normalize(l+v)),0),45)
 *              c = color*(.36+.62*d) + vec3(.25,.38,.43)*rim + vec3(.40)*spec
 *              frag = vec4(pow(c,vec3(.9)),1)
 *
 * Depth test: GL_LESS, depth cleared to 1.0. No face culling (both windings
 * are drawn, matching the unculled GL context). The colour buffer is stored
 * bottom-up, exactly like glReadPixels, because render.py flips it afterwards.
 */
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define NEAR_EPS 1e-6f

typedef struct { float clip[4]; float wp[3]; float n[3]; } Vtx;

static inline void lerp_vtx(Vtx *out, const Vtx *a, const Vtx *b, float t) {
    for (int i = 0; i < 4; i++) out->clip[i] = a->clip[i] + (b->clip[i] - a->clip[i]) * t;
    for (int i = 0; i < 3; i++) out->wp[i] = a->wp[i] + (b->wp[i] - a->wp[i]) * t;
    for (int i = 0; i < 3; i++) out->n[i] = a->n[i] + (b->n[i] - a->n[i]) * t;
}

/* Clip a polygon against the near plane (z + w > 0). */
static int clip_near(const Vtx *in, int n, Vtx *out) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        const Vtx *a = &in[i];
        const Vtx *b = &in[(i + 1) % n];
        float da = a->clip[2] + a->clip[3];
        float db = b->clip[2] + b->clip[3];
        int ina = da > NEAR_EPS, inb = db > NEAR_EPS;
        if (ina) out[m++] = *a;
        if (ina != inb) {
            float t = da / (da - db);
            lerp_vtx(&out[m++], a, b, t);
        }
    }
    return m;
}

static inline float shade_channel(float v) {
    if (v <= 0.f) return 0.f;
    if (v >= 1.f) return 1.f;
    return powf(v, 0.9f);
}

static void shade_pixel(const float *P, const float *N, const float *color,
                        const float *eye, unsigned char *dst) {
    float nx = N[0], ny = N[1], nz = N[2];
    float nl = sqrtf(nx * nx + ny * ny + nz * nz);
    if (nl > 1e-12f) { nx /= nl; ny /= nl; nz /= nl; }

    float lx = -4.f - P[0], ly = 7.f - P[1], lz = 8.f - P[2];
    float ll = sqrtf(lx * lx + ly * ly + lz * lz);
    if (ll > 1e-12f) { lx /= ll; ly /= ll; lz /= ll; }

    float vx = eye[0] - P[0], vy = eye[1] - P[1], vz = eye[2] - P[2];
    float vl = sqrtf(vx * vx + vy * vy + vz * vz);
    if (vl > 1e-12f) { vx /= vl; vy /= vl; vz /= vl; }

    float d = nx * lx + ny * ly + nz * lz;
    if (d < 0.f) d = 0.f;

    float ndv = nx * vx + ny * vy + nz * vz;
    if (ndv < 0.f) ndv = 0.f;
    float t = 1.f - ndv;
    float rim = t * t * t;

    float hx = lx + vx, hy = ly + vy, hz = lz + vz;
    float hl = sqrtf(hx * hx + hy * hy + hz * hz);
    float spec = 0.f;
    if (hl > 1e-12f) {
        float nh = (nx * hx + ny * hy + nz * hz) / hl;
        if (nh > 0.f) spec = powf(nh, 45.f);
    }

    float k = 0.36f + 0.62f * d;
    float c0 = color[0] * k + 0.25f * rim + 0.40f * spec;
    float c1 = color[1] * k + 0.38f * rim + 0.40f * spec;
    float c2 = color[2] * k + 0.43f * rim + 0.40f * spec;

    dst[0] = (unsigned char)(shade_channel(c0) * 255.f + 0.5f);
    dst[1] = (unsigned char)(shade_channel(c1) * 255.f + 0.5f);
    dst[2] = (unsigned char)(shade_channel(c2) * 255.f + 0.5f);
}

/* Rasterize one screen-space triangle (already divided by w). */
static void raster_tri(const Vtx *v0, const Vtx *v1, const Vtx *v2,
                       const float *color, const float *eye,
                       unsigned char *fb, float *depth, int W, int H) {
    float sx[3], sy[3], sz[3], invw[3];
    const Vtx *vs[3] = { v0, v1, v2 };
    float aw[3][3], an[3][3];

    for (int i = 0; i < 3; i++) {
        float w = vs[i]->clip[3];
        if (fabsf(w) < NEAR_EPS) return;
        float iw = 1.f / w;
        sx[i] = (vs[i]->clip[0] * iw * 0.5f + 0.5f) * (float)W;
        sy[i] = (vs[i]->clip[1] * iw * 0.5f + 0.5f) * (float)H;
        sz[i] = vs[i]->clip[2] * iw * 0.5f + 0.5f;
        invw[i] = iw;
        for (int k = 0; k < 3; k++) {
            aw[i][k] = vs[i]->wp[k] * iw;
            an[i][k] = vs[i]->n[k] * iw;
        }
    }

    float area = (sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0]);
    if (fabsf(area) < 1e-9f) return;   /* degenerate */
    float inv_area = 1.f / area;       /* sign handles both windings (no culling) */

    int minx = (int)floorf(fminf(fminf(sx[0], sx[1]), sx[2]));
    int maxx = (int)ceilf(fmaxf(fmaxf(sx[0], sx[1]), sx[2]));
    int miny = (int)floorf(fminf(fminf(sy[0], sy[1]), sy[2]));
    int maxy = (int)ceilf(fmaxf(fmaxf(sy[0], sy[1]), sy[2]));
    if (minx < 0) minx = 0;
    if (miny < 0) miny = 0;
    if (maxx > W - 1) maxx = W - 1;
    if (maxy > H - 1) maxy = H - 1;
    if (minx > maxx || miny > maxy) return;

    for (int y = miny; y <= maxy; y++) {
        float py = (float)y + 0.5f;
        for (int x = minx; x <= maxx; x++) {
            float px = (float)x + 0.5f;
            float w0 = ((sx[1] - px) * (sy[2] - py) - (sx[2] - px) * (sy[1] - py)) * inv_area;
            float w1 = ((sx[2] - px) * (sy[0] - py) - (sx[0] - px) * (sy[2] - py)) * inv_area;
            float w2 = 1.f - w0 - w1;
            if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;

            float z = w0 * sz[0] + w1 * sz[1] + w2 * sz[2];
            if (z < 0.f || z > 1.f) continue;
            long off = (long)y * W + x;
            if (z >= depth[off]) continue;   /* GL_LESS */

            float iw = w0 * invw[0] + w1 * invw[1] + w2 * invw[2];
            if (fabsf(iw) < 1e-20f) continue;
            float rw = 1.f / iw;

            float P[3], N[3];
            for (int k = 0; k < 3; k++) {
                P[k] = (w0 * aw[0][k] + w1 * aw[1][k] + w2 * aw[2][k]) * rw;
                N[k] = (w0 * an[0][k] + w1 * an[1][k] + w2 * an[2][k]) * rw;
            }
            depth[off] = z;
            shade_pixel(P, N, color, eye, fb + off * 3);
        }
    }
}

/* Entry point called from Python via ctypes.
 * clip : nv*4 floats (clip-space positions)
 * wp   : nv*3 floats (world positions)
 * nrm  : nv*3 floats (world normals)
 * idx  : ntris*3 int32 indices
 */
void draw_indexed(const float *clip, const float *wp, const float *nrm,
                  const int *idx, int ntris,
                  const float *color, const float *eye,
                  unsigned char *fb, float *depth, int W, int H) {
    Vtx poly[8], clipped[8], tri[3];
    for (int t = 0; t < ntris; t++) {
        int need_clip = 0;
        for (int i = 0; i < 3; i++) {
            int vi = idx[t * 3 + i];
            const float *c = clip + (long)vi * 4;
            memcpy(tri[i].clip, c, 4 * sizeof(float));
            memcpy(tri[i].wp, wp + (long)vi * 3, 3 * sizeof(float));
            memcpy(tri[i].n, nrm + (long)vi * 3, 3 * sizeof(float));
            if (c[2] + c[3] <= NEAR_EPS) need_clip = 1;
        }
        if (!need_clip) {
            raster_tri(&tri[0], &tri[1], &tri[2], color, eye, fb, depth, W, H);
            continue;
        }
        poly[0] = tri[0]; poly[1] = tri[1]; poly[2] = tri[2];
        int n = clip_near(poly, 3, clipped);
        for (int i = 2; i < n; i++)
            raster_tri(&clipped[0], &clipped[i - 1], &clipped[i], color, eye, fb, depth, W, H);
    }
}

void clear_buffers(unsigned char *fb, float *depth, int W, int H,
                   unsigned char r, unsigned char g, unsigned char b) {
    long n = (long)W * H;
    for (long i = 0; i < n; i++) {
        fb[i * 3 + 0] = r;
        fb[i * 3 + 1] = g;
        fb[i * 3 + 2] = b;
        depth[i] = 1.f;
    }
}
