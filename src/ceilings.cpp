#include <chrono>
#include <cstring>
#include <iomanip>
#include <cmath>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <iostream>
#include <omp.h>
#include <string>
#include "platform.h"

using Clock = std::chrono::steady_clock;
constexpr auto duration = std::chrono::milliseconds(350);

void measure_copy(int threads, size_t total_bytes) {
    const size_t length = total_bytes / (2 * size_t(threads)) & ~size_t(63);
    if (length < 64) throw std::runtime_error("Copy buffers are too small");
    const size_t actual_bytes = 2 * size_t(threads) * length;
    std::vector<unsigned char> first(actual_bytes / 2, 1);
    std::vector<unsigned char> second(actual_bytes / 2, 2);
    const int copies_per_batch = length <= 1024 * 1024 ? 16 : 1;
    for (int repeat = 0; repeat < 3; ++repeat) {
        uint64_t traffic = 0;
        unsigned checksum = 0;
        Clock::time_point start;
        #pragma omp parallel num_threads(threads) reduction(+:traffic,checksum)
        {
            const size_t offset = size_t(omp_get_thread_num()) * length;
            auto *src = first.data() + offset;
            auto *dst = second.data() + offset;
            std::memcpy(dst, src, length);
            #pragma omp barrier
            #pragma omp single
            start = Clock::now();
            do {
                for (int i = 0; i < copies_per_batch; ++i) {
                    std::memcpy(dst, src, length);
                    std::swap(src, dst);
                }
                traffic += 2 * length * copies_per_batch;
            } while (Clock::now() - start < duration);
            checksum += src[length - 1];
        }
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        if (checksum != unsigned(threads)) throw std::runtime_error("Unexpected copy checksum");
        std::cout << "copy," << threads << ',' << actual_bytes << ',' << repeat << ','
                  << traffic << ',' << seconds << ',' << traffic / seconds / 1e9 << '\n';
    }
}

void measure_compute(int threads) {

    for (int repeat = 0; repeat < 3; ++repeat) {
        uint64_t flops = 0;
        double checksum = 0;
        Clock::time_point start;
        #pragma omp parallel num_threads(threads) reduction(+:flops,checksum)
        {
#if defined(__AVX2__) && defined(__FMA__)
            using Vector = __m256;
            constexpr int lanes = 8;
            const auto multiplier = _mm256_set1_ps(0.999999f);
            const auto increment = _mm256_set1_ps(0.000001f);
            auto broadcast = [](float x) { return _mm256_set1_ps(x); };
            auto fma = [&](Vector x) { return _mm256_fmadd_ps(x, multiplier, increment); };
            auto lane = [](Vector x) { return _mm_cvtss_f32(_mm256_castps256_ps128(x)); };
#elif defined(__aarch64__)
            using Vector = float32x4_t;
            constexpr int lanes = 4;
            const auto multiplier = vdupq_n_f32(0.999999f);
            const auto increment = vdupq_n_f32(0.000001f);
            auto broadcast = [](float x) { return vdupq_n_f32(x); };
            auto fma = [&](Vector x) { return vfmaq_f32(increment, x, multiplier); };
            auto lane = [](Vector x) { return vgetq_lane_f32(x, 0); };
#else
            using Vector = float;
            constexpr int lanes = 1;
            auto broadcast = [](float x) { return x; };
            auto fma = [](float x) { return std::fma(x, 0.999999f, 0.000001f); };
            auto lane = [](float x) { return x; };
#endif
            Vector accumulators[8];
            for (int i = 0; i < 8; ++i) accumulators[i] = broadcast(0.1f * i);
            #pragma omp single
            start = Clock::now();
            do {
                for (int batch = 0; batch < 4096; ++batch)
                    for (auto &value : accumulators) value = fma(value);
                flops += 4096 * 8 * lanes * 2;
            } while (Clock::now() - start < duration);
            for (auto value : accumulators) checksum += lane(value);
        }
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        if (checksum <= 0) throw std::runtime_error("Unexpected FMA checksum");
        std::cout << "fma," << threads << ",0," << repeat << ',' << flops << ','
                  << seconds << ',' << flops / seconds / 1e9 << '\n';
    }
}

int main(int argc, char **argv) try {
    if (argc < 3 || argc > 4) throw std::runtime_error("Expected copy threads total_bytes or fma threads");
    const std::string mode = argv[1];
    const int threads = std::stoi(argv[2]);
    if (threads < 1 || threads > 512) throw std::runtime_error("Expected 1..512 threads");
    pin_performance_cores(true, size_t(threads));
    omp_set_dynamic(0);
    std::cout << std::setprecision(10);
    std::cout << "mode,threads,total_buffer_bytes,repeat,work,seconds,giga_per_second\n";
    if (mode == "copy" && argc == 4) {
        const size_t bytes = std::stoull(argv[3]);
        if (bytes > 256 * 1024 * 1024) throw std::runtime_error("Maximum copy footprint is 256 MiB");
        measure_copy(threads, bytes);
    } else if (mode == "fma" && argc == 3) {
        measure_compute(threads);
    } else {
        throw std::runtime_error("Expected copy threads total_bytes or fma threads");
    }
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
