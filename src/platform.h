#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
inline unsigned efficiency_class(const PROCESSOR_RELATIONSHIP &p) {
    // Windows ABI puts EfficiencyClass at byte 1. Older MinGW headers name
    // that byte Reserved[0], while Windows 11 fills it using the current ABI.
    return reinterpret_cast<const BYTE *>(&p)[1];
}
inline std::vector<ULONG_PTR> performance_cores() {
    DWORD size = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &size);
    std::vector<char> data(size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(data.data()), &size))
        throw std::runtime_error("Cannot query CPU topology");
    std::vector<std::pair<unsigned, ULONG_PTR>> cores;
    unsigned best = 0;
    for (DWORD offset = 0; offset < size;) {
        auto *info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(data.data() + offset);
        auto &p = info->Processor;
        if (p.GroupCount != 1 || p.GroupMask[0].Group != 0)
            throw std::runtime_error("This Windows driver supports one processor group");
        best = std::max(best, efficiency_class(p));
        auto mask = p.GroupMask[0].Mask;
        cores.push_back({efficiency_class(p), mask & (~mask + 1)});
        offset += info->Size;
    }
    std::vector<ULONG_PTR> result;
    for (auto core : cores) if (core.first == best) result.push_back(core.second);
    return result;
}
inline void pin_performance_cores(bool enabled = true, size_t count = 0) {
    if (!enabled) return;
    ULONG_PTR mask = 0;
    auto cores = performance_cores();
    if (count > cores.size()) throw std::runtime_error("Requested more performance cores than available");
    if (count) cores.resize(count);
    for (auto core : cores) mask |= core;
    if (!SetProcessAffinityMask(GetCurrentProcess(), mask))
        throw std::runtime_error("Cannot set performance-core affinity");
}
inline uint64_t peak_rss() {
    PROCESS_MEMORY_COUNTERS info{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info)))
        throw std::runtime_error("Cannot read peak working set");
    return info.PeakWorkingSetSize;
}
inline uint64_t free_ram() {
    MEMORYSTATUSEX info{};
    info.dwLength = sizeof(info);
    GlobalMemoryStatusEx(&info);
    return info.ullAvailPhys;
}
#else
#include <sys/resource.h>
#include <unistd.h>
inline void pin_performance_cores(bool = true, size_t = 0) {}
inline uint64_t peak_rss() { rusage r{}; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss * 1024ULL; }
inline uint64_t free_ram() { return sysconf(_SC_AVPHYS_PAGES) * uint64_t(sysconf(_SC_PAGESIZE)); }
#endif
