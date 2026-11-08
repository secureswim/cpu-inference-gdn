#include <algorithm>
#include <iostream>
#include <cpuid.h>
#include "platform.h"
int main() {
    char brand[49]{};
    for (unsigned i = 0; i < 3; ++i) {
        unsigned a,b,c,d;
        __cpuid(0x80000002+i,a,b,c,d);
        auto p = reinterpret_cast<unsigned *>(brand + i*16);
        p[0]=a; p[1]=b; p[2]=c; p[3]=d;
    }
    __builtin_cpu_init();
    std::cout << "cpu=" << brand << "\navx2=" << bool(__builtin_cpu_supports("avx2"))
              << "\navx512f=" << bool(__builtin_cpu_supports("avx512f")) << "\n";
#ifdef _WIN32
    MEMORYSTATUSEX m{}; m.dwLength=sizeof(m); GlobalMemoryStatusEx(&m);
    SYSTEM_POWER_STATUS power{}; GetSystemPowerStatus(&power);
    std::cout << "ram_bytes=" << m.ullTotalPhys << "\nfree_ram_bytes=" << m.ullAvailPhys
              << "\nac_line_status=" << int(power.ACLineStatus) << "\np_core_masks=";
    for(auto mask: performance_cores()) std::cout << mask << ",";
    std::cout << "\n";
    DWORD bytes=0;
    GetLogicalProcessorInformationEx(RelationAll,nullptr,&bytes);
    std::vector<char> data(bytes);
    GetLogicalProcessorInformationEx(RelationAll,
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(data.data()),&bytes);
    for(DWORD offset=0; offset<bytes;) {
        auto *i=reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(data.data()+offset);
        if(i->Relationship==RelationProcessorCore)
            std::cout << "core=" << efficiency_class(i->Processor) << "," << i->Processor.GroupMask[0].Mask << "\n";
        if(i->Relationship==RelationCache)
            std::cout << "cache=" << unsigned(i->Cache.Level) << "," << i->Cache.CacheSize << "," << unsigned(i->Cache.Type) << "\n";
        offset+=i->Size;
    }
#endif
}
