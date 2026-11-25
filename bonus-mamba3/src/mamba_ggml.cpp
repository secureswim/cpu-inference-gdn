// Stretch experiment: a stateless ggml custom node consumes a packed input
// tensor and emits a packed next-state tensor. No model loader changes.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include "ggml.h"
#include "ggml-cpu.h"
#include "mamba.h"
struct Shape {
    int h=2,p=13,n=30,r=1;
    std::array<size_t,18> offsets{};
    size_t count=0;
    explicit Shape(int rank):r(rank) {
        std::array<size_t,18> lengths={size_t(h*p*n),size_t(r*h*n),size_t(h*p),size_t(h*(n/4)),
            size_t(r*h*n),size_t(r*h*n),size_t(h*p),size_t(h),size_t(h),size_t(h),
            size_t(r*h*p),size_t(r*h*p),size_t(h*p),size_t(r*h*p),size_t(h),
            size_t(h*(n/4)),size_t(h*p),size_t(h*3*r*n)};
        for(size_t i=0;i<18;++i) {offsets[i]=count;count+=lengths[i];}
    }
};
void run(float *data,const Shape &s) {
    std::array<float *,18> p;
    for(size_t i=0;i<18;++i) p[i]=data+s.offsets[i];
    if(mamba_step(s.h,s.p,s.n,s.r,3,1,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],
        p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17]))
        throw std::runtime_error("Kernel failed");
}
void custom(ggml_tensor *dst,const ggml_tensor *src,int ith,int nth,void *userdata) {
    GGML_ASSERT(ith==0 && nth==1);
    auto &shape=*static_cast<Shape *>(userdata);
    std::memcpy(dst->data,src->data,shape.count*4);
    run(static_cast<float *>(dst->data),shape);
}
int main() {
    for(int rank:{1,4}) {
        Shape shape(rank);
        std::mt19937 rng(97);
        std::uniform_real_distribution<float> dist(-0.1f,0.1f);
        std::vector<float> expected(shape.count);
        for(auto &v:expected) v=dist(rng);
        for(int h=0;h<shape.h;++h) {
            expected[shape.offsets[7]+h]=-1;
            expected[shape.offsets[8]+h]=0.05f;
            expected[shape.offsets[9]+h]=0.6f;
        }
        auto ctx=ggml_init({4*1024*1024,nullptr,false});
        auto input=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,shape.count);
        auto output=ggml_map_custom1(ctx,input,custom,1,&shape);
        auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,output);
        auto plan=ggml_graph_plan(graph,1,nullptr);
        std::vector<uint8_t> work(plan.work_size);plan.work_data=work.data();
        std::memcpy(input->data,expected.data(),shape.count*4);
        double error=0;
        for(int step=0;step<1000;++step) {
            run(expected.data(),shape);
            if(ggml_graph_compute(graph,&plan)!=GGML_STATUS_SUCCESS) return 1;
            auto actual=static_cast<float *>(output->data);
            for(size_t i=0;i<shape.count;++i) error=std::max(error,double(std::abs(actual[i]-expected[i])));
            std::memcpy(input->data,actual,shape.count*4);
        }
        std::cout << "rank=" << rank << ",steps=1000,max_abs=" << error << "\n";
        ggml_free(ctx);
        if(error!=0) return 1;
    }
}
