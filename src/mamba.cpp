// Mamba-2/3 recurrence kernels. State layout [head, value, state_dimension].
// Mamba-3 equations follow the official FP32/FP64 PyTorch reference at
// state-spaces/mamba e9594ce1c732d97440f0332fdc43170a2294dbfa.
// Dense input/output projections and BC normalization precede/follow this API.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

static float sum8(__m256 value) {
    __m128 sum=_mm_add_ps(_mm256_castps256_ps128(value),_mm256_extractf128_ps(value,1));
    sum=_mm_hadd_ps(sum,sum); sum=_mm_hadd_ps(sum,sum);
    return _mm_cvtss_f32(sum);
}

// Explicitly prevent automatic vectorization in the plain C++ baseline.
__attribute__((optimize("no-tree-vectorize")))
static void scalar_row(int n,int rank,float *s,const float *b,const float *c,
                       const float *previous,const float *current,
                       float alpha,float beta,float gamma,float *outputs) {
    for(int i=0;i<n;++i) {
        float value=s[i]*alpha;
        // Match the reference's separate current and previous rank reductions.
        float curr=0,prev=0;
        for(int r=0;r<rank;++r) {
            curr+=(gamma*current[r])*b[r*n+i];
            prev+=(beta*previous[r])*b[(rank+r)*n+i];
        }
        s[i]=(value+curr)+prev;
    }
    for(int r=0;r<rank;++r) {
        float dot=0;
        for(int i=0;i<n;++i) dot+=s[i]*c[r*n+i];
        outputs[r]=dot;
    }
}

static void avx_row(int n,int rank,float *s,const float *b,const float *c,
                    const float *previous,const float *current,
                    float alpha,float beta,float gamma,float *outputs) {
    __m256 acc[4]={_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps()};
    int i=0;
    for(;i+8<=n;i+=8) {
        __m256 value=_mm256_mul_ps(_mm256_loadu_ps(s+i),_mm256_set1_ps(alpha));
        __m256 curr=_mm256_setzero_ps(),prev=_mm256_setzero_ps();
        for(int r=0;r<rank;++r) {
            curr=_mm256_fmadd_ps(_mm256_set1_ps(gamma*current[r]),_mm256_loadu_ps(b+r*n+i),curr);
            prev=_mm256_fmadd_ps(_mm256_set1_ps(beta*previous[r]),_mm256_loadu_ps(b+(rank+r)*n+i),prev);
        }
        value=_mm256_add_ps(_mm256_add_ps(value,curr),prev);
        _mm256_storeu_ps(s+i,value);
        // Readout immediately reuses the updated state in registers.
        for(int r=0;r<rank;++r) acc[r]=_mm256_fmadd_ps(value,_mm256_loadu_ps(c+r*n+i),acc[r]);
    }
    for(int r=0;r<rank;++r) outputs[r]=sum8(acc[r]);
    for(;i<n;++i) {
        float curr=0,prev=0;
        for(int r=0;r<rank;++r) {
            curr+=(gamma*current[r])*b[r*n+i];
            prev+=(beta*previous[r])*b[(rank+r)*n+i];
        }
        s[i]=(s[i]*alpha+curr)+prev;
        for(int r=0;r<rank;++r) outputs[r]+=s[i]*c[r*n+i];
    }
}

__attribute__((optimize("no-tree-vectorize")))
static float mamba2_scalar(int n,float *s,const float *b,const float *c,float alpha,float dx) {
    float sum=0;
    for(int i=0;i<n;++i) { s[i]=alpha*s[i]+dx*b[i]; sum+=s[i]*c[i]; }
    return sum;
}

static float mamba2_avx(int n,float *s,const float *b,const float *c,float alpha,float dx) {
    __m256 total=_mm256_setzero_ps();
    int i=0;
    for(;i+8<=n;i+=8) {
        auto value=_mm256_fmadd_ps(_mm256_set1_ps(dx),_mm256_loadu_ps(b+i),
            _mm256_mul_ps(_mm256_set1_ps(alpha),_mm256_loadu_ps(s+i)));
        _mm256_storeu_ps(s+i,value);
        total=_mm256_fmadd_ps(value,_mm256_loadu_ps(c+i),total);
    }
    float sum=sum8(total);
    for(;i<n;++i) { s[i]=alpha*s[i]+dx*b[i]; sum+=s[i]*c[i]; }
    return sum;
}

// Inputs are contiguous, batch size one. r must be 1 or 4, variant 2 or 3.
// scratch holds H*(3*R*N) floats, without allocations inside the decode loop.
// B/C already include normalization and biases. angle_proj is pre-tanh angular
// velocity, as in the official rotary reference. DT/TRAP are post-softplus/sigmoid.
EXPORT int mamba_step(int heads,int p,int n,int rank,int variant,int simd,
    float *state,float *bprev,float *xprev,float *angle,
    const float *b,const float *c,const float *x,const float *a,const float *dt,
    const float *trap,const float *xproj,const float *oproj,
    const float *z,const float *zproj,const float *skip,const float *angle_proj,
    float *output,float *scratch) {
    if(heads<1 || p<1 || n<2 || n%2 || (rank!=1 && rank!=4) ||
       (variant!=2 && variant!=3) || (variant==2 && rank!=1)) return 1;
    const int angles=n/4;
    for(int h=0;h<heads;++h) {
        float *bk=scratch+h*3*rank*n;
        float *cq=bk+2*rank*n;
        for(int r=0;r<rank;++r) {
            const int src=(r*heads+h)*n;
            std::memcpy(bk+r*n,b+src,n*4);
            std::memcpy(bk+(rank+r)*n,bprev+src,n*4);
            std::memcpy(cq+r*n,c+src,n*4);
        }
        if(variant==3) {
            for(int j=0;j<angles;++j) {
                float &theta=angle[h*angles+j];
                theta+=std::tanh(angle_proj[h*angles+j])*dt[h]*float(3.141592653589793);
                float cs=std::cos(theta),sn=std::sin(theta);
                int lo=rank==1 ? 2*j : j;
                int hi=rank==1 ? 2*j+1 : j+n/2;
                for(int r=0;r<rank;++r) for(float *t:{bk+r*n,cq+r*n}) {
                    float v0=t[lo],v1=t[hi];
                    t[lo]=v0*cs-v1*sn; t[hi]=v0*sn+v1*cs;
                }
            }
        }
        float alpha=std::exp(a[h]*dt[h]);
        float beta=variant==3 ? (1-trap[h])*dt[h]*alpha : 0;
        float gamma=variant==3 ? trap[h]*dt[h] : dt[h];
        for(int j=0;j<p;++j) {
            float current[4]{},previous[4]{},dots[4]{};
            for(int r=0;r<rank;++r) {
                float xp=xproj[(r*heads+h)*p+j];
                current[r]=x[h*p+j]*xp;
                previous[r]=xprev[h*p+j]*xp;
            }
            auto row=state+(h*p+j)*n;
            if(variant==2) dots[0]=simd ? mamba2_avx(n,row,bk,cq,alpha,gamma*current[0])
                                      : mamba2_scalar(n,row,bk,cq,alpha,gamma*current[0]);
            else if(simd) avx_row(n,rank,row,bk,cq,previous,current,alpha,beta,gamma,dots);
            else scalar_row(n,rank,row,bk,cq,previous,current,alpha,beta,gamma,dots);
            float y=0;
            for(int r=0;r<rank;++r) {
                float zz=z[h*p+j]*zproj[(r*heads+h)*p+j];
                float gate=zz/(1+std::exp(-zz));
                y+=(dots[r]+skip[h]*current[r])*gate*oproj[(r*heads+h)*p+j];
            }
            output[h*p+j]=y;
        }
        for(int r=0;r<rank;++r) std::memcpy(bprev+(r*heads+h)*n,bk+r*n,n*4);
    }
    std::memcpy(xprev,x,heads*p*4);
    return 0;
}

EXPORT double mamba_bench(int iterations,int heads,int p,int n,int rank,int variant,int simd,
    float *state,float *bprev,float *xprev,float *angle,
    const float *b,const float *c,const float *x,const float *a,const float *dt,
    const float *trap,const float *xproj,const float *oproj,
    const float *z,const float *zproj,const float *skip,const float *angle_proj,
    float *output,float *scratch) {
    auto start=std::chrono::steady_clock::now();
    for(int i=0;i<iterations;++i) {
        if(mamba_step(heads,p,n,rank,variant,simd,state,bprev,xprev,angle,
            b,c,x,a,dt,trap,xproj,oproj,z,zproj,skip,angle_proj,output,scratch)) return -1;
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/iterations;
}
