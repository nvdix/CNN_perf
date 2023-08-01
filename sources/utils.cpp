#include <cpuid.h>

#include "headers/consts.h"
#include "headers/utils.h"

// Форматирование в строку std::string с праметрами
std::string string_format(const std::string fmt_str, ...)
{
    int final_n, n = ((int)fmt_str.size()) * 2; /* Reserve two times as much as the length of the fmt_str */
    std::unique_ptr<char[]> formatted;
    va_list ap;
    while(1) {
        formatted.reset(new char[n]); /* Wrap the plain char array into the unique_ptr */
        strcpy(&formatted[0], fmt_str.c_str());
        va_start(ap, fmt_str);
        final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
        va_end(ap);
        if (final_n < 0 || final_n >= n)
            n += abs(final_n - n + 1);
        else
            break;
    }
    return std::string(formatted.get());
}

void cpuid(int info[4], int InfoType)
{
    __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}

// Проверка наличия указанного вида оптимизации в процессоре
bool CheckOptimizationCPU(OPT_TYPE opt)
{
    //  Misc.
    bool HW_MMX = false;
    bool HW_x64 = false;
    bool HW_ABM = false;      // Advanced Bit Manipulation
    bool HW_RDRAND = false;
    bool HW_BMI1 = false;
    bool HW_BMI2 = false;
    bool HW_ADX = false;
    bool HW_PREFETCHWT1 = false;

    //  SIMD: 128-bit
    bool HW_SSE = false;
    bool HW_SSE2 = false;
    bool HW_SSE3 = false;
    bool HW_SSSE3 = false;
    bool HW_SSE41 = false;
    bool HW_SSE42 = false;
    bool HW_SSE4a = false;
    bool HW_AES = false;
    bool HW_SHA = false;

    //  SIMD: 256-bit
    bool HW_AVX = false;
    bool HW_XOP = false;
    bool HW_FMA3 = false;
    bool HW_FMA4 = false;
    bool HW_AVX2 = false;

    //  SIMD: 512-bit
    bool HW_AVX512F = false;    //  AVX512 Foundation
    bool HW_AVX512CD = false;   //  AVX512 Conflict Detection
    bool HW_AVX512PF = false;   //  AVX512 Prefetch
    bool HW_AVX512ER = false;   //  AVX512 Exponential + Reciprocal
    bool HW_AVX512VL = false;   //  AVX512 Vector Length Extensions
    bool HW_AVX512BW = false;   //  AVX512 Byte + Word
    bool HW_AVX512DQ = false;   //  AVX512 Doubleword + Quadword
    bool HW_AVX512IFMA = false; //  AVX512 Integer 52-bit Fused Multiply-Add
    bool HW_AVX512VBMI = false; //  AVX512 Vector Byte Manipulation Instructions

    int info[4];
    cpuid(info, 0);
    int nIds = info[0];

    cpuid(info, 0x80000000);
    unsigned nExIds = info[0];

    //  Detect Features
    if (nIds >= 0x00000001)
    {
        cpuid(info,0x00000001);
        HW_MMX    = (info[3] & ((int)1 << 23)) != 0;
        HW_SSE    = (info[3] & ((int)1 << 25)) != 0;
        HW_SSE2   = (info[3] & ((int)1 << 26)) != 0;
        HW_SSE3   = (info[2] & ((int)1 <<  0)) != 0;

        HW_SSSE3  = (info[2] & ((int)1 <<  9)) != 0;
        HW_SSE41  = (info[2] & ((int)1 << 19)) != 0;
        HW_SSE42  = (info[2] & ((int)1 << 20)) != 0;
        HW_AES    = (info[2] & ((int)1 << 25)) != 0;

        HW_AVX    = (info[2] & ((int)1 << 28)) != 0;
        HW_FMA3   = (info[2] & ((int)1 << 12)) != 0;

        HW_RDRAND = (info[2] & ((int)1 << 30)) != 0;
    }

    if (nIds >= 0x00000007)
    {
        cpuid(info,0x00000007);
        HW_AVX2   = (info[1] & ((int)1 <<  5)) != 0;

        HW_BMI1        = (info[1] & ((int)1 <<  3)) != 0;
        HW_BMI2        = (info[1] & ((int)1 <<  8)) != 0;
        HW_ADX         = (info[1] & ((int)1 << 19)) != 0;
        HW_SHA         = (info[1] & ((int)1 << 29)) != 0;
        HW_PREFETCHWT1 = (info[2] & ((int)1 <<  0)) != 0;

        HW_AVX512F     = (info[1] & ((int)1 << 16)) != 0;
        HW_AVX512CD    = (info[1] & ((int)1 << 28)) != 0;
        HW_AVX512PF    = (info[1] & ((int)1 << 26)) != 0;
        HW_AVX512ER    = (info[1] & ((int)1 << 27)) != 0;
        HW_AVX512VL    = (info[1] & ((int)1 << 31)) != 0;
        HW_AVX512BW    = (info[1] & ((int)1 << 30)) != 0;
        HW_AVX512DQ    = (info[1] & ((int)1 << 17)) != 0;
        HW_AVX512IFMA  = (info[1] & ((int)1 << 21)) != 0;
        HW_AVX512VBMI  = (info[2] & ((int)1 <<  1)) != 0;
    }

    if (nExIds >= 0x80000001)
    {
        cpuid(info,0x80000001);
        HW_x64   = (info[3] & ((int)1 << 29)) != 0;
        HW_ABM   = (info[2] & ((int)1 <<  5)) != 0;
        HW_SSE4a = (info[2] & ((int)1 <<  6)) != 0;
        HW_FMA4  = (info[2] & ((int)1 << 16)) != 0;
        HW_XOP   = (info[2] & ((int)1 << 11)) != 0;
    }
    if (opt == OPT_TYPE::SSE && HW_SSE2 && HW_SSSE3 && HW_SSE3 && HW_SSE41 && HW_SSE42)
        return true;

    if (opt == OPT_TYPE::AVX && HW_AVX && HW_AVX2 && HW_FMA3)
        return true;

    if (opt == OPT_TYPE::AVX512 && HW_AVX512F && HW_AVX512BW)
        return true;

    if (opt == OPT_TYPE::NONE)
        return true;

    return false;
}

// Получение текстовой информации о процессоре
std::string GetCpuInfo()
 {
     // 4 is essentially hardcoded due to the __cpuid function requirements.
     // NOTE: Results are limited to whatever the sizeof(int) * 4 is...
     std::array<int, 4> integerBuffer = {};
     constexpr size_t sizeofIntegerBuffer = sizeof(int) * integerBuffer.size();

     std::array<char, 64> charBuffer = {};

     // The information you wanna query __cpuid for.
     // https://learn.microsoft.com/en-us/cpp/intrinsics/cpuid-cpuidex?view=vs-2019
     constexpr std::array<unsigned int, 3> functionIds = {
         // Manufacturer
         //  EX: "Intel(R) Core(TM"
         0x80000002,
         // Model
         //  EX: ") i7-8700K CPU @"
         0x80000003,
         // Clockspeed
         //  EX: " 3.70GHz"
         0x80000004
     };

     std::string cpu;

     for (int id : functionIds)
     {
         // Get the data for the current ID.
         cpuid(integerBuffer.data(), id);

         // Copy the raw data from the integer buffer into the character buffer
         std::memcpy(charBuffer.data(), integerBuffer.data(), sizeofIntegerBuffer);

         // Copy that data into a std::string
         cpu += std::string(charBuffer.data());
     }

     return cpu;
 }
