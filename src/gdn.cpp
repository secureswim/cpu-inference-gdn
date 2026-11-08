#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "ggml.h"
#include "ggml-cpu.h"
#include "platform.h"
using Clock = std::chrono::steady_clock;
extern "C" void project_profile_enable(int);
extern "C" double project_profile_read(int);
constexpr int D = 128, HK = 16, HV = 32;
constexpr size_t STATE = size_t(D)*D*HV;

struct Graph {
    ggml_context *ctx;
    ggml_tensor *q, *k, *v, *g, *beta, *state, *out;
    ggml_cgraph *graph;
    ggml_threadpool *pool;
    ggml_cplan plan;
    std::vector<uint8_t> work;
    int tokens;
    Graph(int n, int threads): tokens(n) {
        size_t floats = size_t(n)*(2*HK*D+2*HV*D+2*HV)+2*STATE;
        ctx=ggml_init({floats*sizeof(float)+4*1024*1024,nullptr,false});
        if (!ctx) throw std::runtime_error("ggml allocation failed");
        q=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,D,HK,n,1);
        k=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,D,HK,n,1);
        v=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,D,HV,n,1);
        g=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,1,HV,n,1);
        beta=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,1,HV,n,1);
        state=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,D,D,HV,1);
        out=ggml_gated_delta_net(ctx,q,k,v,g,beta,state,1);
        graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,out);
        auto params=ggml_threadpool_params_default(threads);
        params.poll=0;
        pool=ggml_threadpool_new(&params);
        plan=ggml_graph_plan(graph,threads,pool);
        work.resize(plan.work_size); plan.work_data=work.data();
        std::mt19937 rng(2023522);
        std::normal_distribution<float> normal(0,1);
        for(auto t : {q,k,v}) {
            auto data=static_cast<float *>(t->data);
            for(int64_t i=0; i<ggml_nelements(t); ++i) data[i]=normal(rng);
        }
        for(auto t : {q,k}) {
            auto data=static_cast<float *>(t->data);
            for(int row=0; row<n*HK; ++row) {
                double norm=0;
                for(int d=0;d<D;++d) norm+=double(data[row*D+d])*data[row*D+d];
                for(int d=0;d<D;++d) data[row*D+d]/=float(std::sqrt(norm));
            }
        }
        std::fill_n(static_cast<float *>(g->data),n*HV,-0.05f);
        std::fill_n(static_cast<float *>(beta->data),n*HV,0.3f);
        reset();
    }
    ~Graph() { ggml_threadpool_free(pool); ggml_free(ctx); }
    void reset() { std::memset(state->data,0,STATE*4); }
    void compute() {
        if (ggml_graph_compute(graph,&plan)!=GGML_STATUS_SUCCESS)
            throw std::runtime_error("ggml graph failed");
    }
    void advance() {
        compute();
        std::memcpy(state->data,static_cast<float *>(out->data)+tokens*HV*D,STATE*4);
    }
};

void check() {
    Graph a(1,1);
    std::mt19937 rng(97);
    std::uniform_real_distribution<float> dist(-0.2f,0.2f);
    std::vector<double> ref(STATE);
    for(size_t i=0;i<STATE;++i) ref[i]=static_cast<float *>(a.state->data)[i]=dist(rng);
    double max_state=0,max_output=0;
    for(int step=0;step<1000;++step) {
        for(auto t:{a.q,a.k,a.v}) {
            auto p=static_cast<float *>(t->data);
            for(int64_t i=0;i<ggml_nelements(t);++i) p[i]=dist(rng);
        }
        a.compute();
        for(int h=0;h<HV;++h) {
            auto q=static_cast<float *>(a.q->data)+(h%HK)*D;
            auto k=static_cast<float *>(a.k->data)+(h%HK)*D;
            auto v=static_cast<float *>(a.v->data)+h*D;
            const double decay=std::exp(double(static_cast<float *>(a.g->data)[h]));
            for(int j=0;j<D;++j) {
                auto row=ref.data()+(h*D+j)*D;
                double dot=0;
                for(int i=0;i<D;++i) { row[i]*=decay; dot+=row[i]*k[i]; }
                const double delta=(v[j]-dot)*double(static_cast<float *>(a.beta->data)[h]);
                double y=0;
                for(int i=0;i<D;++i) { row[i]+=delta*k[i]; y+=row[i]*q[i]; }
                max_output=std::max(max_output,std::abs(y/std::sqrt(double(D))-static_cast<float *>(a.out->data)[h*D+j]));
            }
        }
        auto s=static_cast<float *>(a.out->data)+HV*D;
        for(size_t i=0;i<STATE;++i) max_state=std::max(max_state,std::abs(ref[i]-s[i]));
        std::memcpy(a.state->data,s,STATE*4);
    }
    // A multi-token graph must equal repeated one-token graphs, including broadcasting.
    Graph sequence(17,1); Graph recurrent(1,1);
    sequence.compute();
    double max_chunk=0;
    for(int t=0;t<17;++t) {
        ggml_tensor *dst[]={recurrent.q,recurrent.k,recurrent.v,recurrent.g,recurrent.beta};
        ggml_tensor *src[]={sequence.q,sequence.k,sequence.v,sequence.g,sequence.beta};
        for(int j=0;j<5;++j) std::memcpy(dst[j]->data,static_cast<char *>(src[j]->data)+t*src[j]->nb[2],dst[j]->nb[2]);
        recurrent.advance();
        for(int i=0;i<HV*D;++i)
            max_chunk=std::max(max_chunk,double(std::abs(static_cast<float *>(recurrent.out->data)[i]-static_cast<float *>(sequence.out->data)[t*HV*D+i])));
    }
    auto final_state=static_cast<float *>(sequence.out->data)+17*HV*D;
    for(size_t i=0;i<STATE;++i) max_chunk=std::max(max_chunk,double(std::abs(final_state[i]-static_cast<float *>(recurrent.state->data)[i])));
    std::cout << "{\"steps\":1000,\"max_state_abs\":" << max_state
              << ",\"max_output_abs\":" << max_output << ",\"chunk_abs\":" << max_chunk << "}\n";
    if(max_state>2e-5 || max_output>2e-5 || max_chunk>2e-5) throw std::runtime_error("Correctness failed");
}

int main(int argc,char **argv) try {
    if(argc<2) throw std::runtime_error("Usage: gdn check | bench threads prefill|decode|profile length pinned");
    pin_performance_cores(argc<6 || std::stoi(argv[5])!=0);
    std::cout << std::setprecision(10);
    if(std::string(argv[1])=="check") { check(); return 0; }
    if(argc!=6) throw std::runtime_error("Expected bench threads mode length pinned");
    if(std::string(argv[1])!="bench") throw std::runtime_error("Unknown command");
    int threads=std::stoi(argv[2]), length=std::stoi(argv[4]);
    std::string mode=argv[3];
    if(mode!="prefill" && mode!="decode" && mode!="profile" && mode!="profile_prefill")
        throw std::runtime_error("Unknown benchmark mode");
    if(threads<1 || threads>512 || length<128 || length>32768 || length%128)
        throw std::runtime_error("Use 1..512 threads and a length divisible by 128, from 128 to 32768");
    if(std::string(argv[5])!="0" && std::string(argv[5])!="1")
        throw std::runtime_error("Pinned must be 0 or 1");
    if(free_ram()<1000000000ULL) throw std::runtime_error("Insufficient free RAM");
    Graph decode(1,threads), chunk(128,threads);
    if(mode=="decode") {
        for(int t=0;t<length;t+=128) chunk.advance();
        std::memcpy(decode.state->data,chunk.state->data,STATE*4);
    }
    const bool prefill=mode=="prefill" || mode=="profile_prefill";
    const bool profile=mode=="profile" || mode=="profile_prefill";
    Graph &a=prefill ? chunk : decode;
    std::vector<float> initial(STATE);
    std::memcpy(initial.data(),a.state->data,STATE*4);
    a.advance();
    for(int repeat=0;repeat<3;++repeat) {
        int calls=0;
        double elapsed=0;
        std::memcpy(a.state->data,initial.data(),STATE*4);
        auto start=Clock::now();
        if(profile) project_profile_enable(1);
        do {
            if(prefill) a.reset();
            int blocks=prefill ? length/128 : 1;
            for(int j=0;j<blocks;++j) a.advance();
            ++calls;
            elapsed=std::chrono::duration<double>(Clock::now()-start).count();
        } while(elapsed<0.35);
        project_profile_enable(0);
        double tokens=double(calls)*(prefill ? length : 1);
        std::cout << mode << "," << length << "," << threads << "," << argv[5] << ","
                  << repeat << "," << tokens << "," << elapsed << "," << tokens/elapsed
                  << "," << peak_rss() << "," << free_ram();
        for(int p=0;p<5;++p) std::cout << "," << project_profile_read(p);
        std::cout << "\n";
    }
    return 0;
} catch(const std::exception &e) { std::cerr << e.what() << "\n"; return 1; }
