// Minimal float ops for spectral MLP deploy.
#ifndef NN_OPS_FLOAT_H
#define NN_OPS_FLOAT_H

static inline void relu(float *x, int n)
{
    for (int i = 0; i < n; ++i) {
        x[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

#endif /* NN_OPS_FLOAT_H */
