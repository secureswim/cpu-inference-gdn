#pragma once
extern "C" int mamba_step(int heads,int p,int n,int rank,int variant,int simd,
    float *state,float *bprev,float *xprev,float *angle,
    const float *b,const float *c,const float *x,const float *a,const float *dt,
    const float *trap,const float *xproj,const float *oproj,
    const float *z,const float *zproj,const float *skip,const float *angle_proj,
    float *output,float *scratch);
