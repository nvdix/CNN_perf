#ifndef MULSUM_H
#define MULSUM_H

#include <fstream>
#include <vector>
#include <typeinfo>
#include <stdexcept>

#include "../baseconv.h"

// Умножение и суммирование целочисленных 8-разрядных операндов (AVX512-512 бит)
extern __attribute__((target("avx512f,avx512bw"))) inline void dot_product_optimized_AVX512(char* a_ptr, char* w_ptr, size_t n, char& sum_val);
// Умножение и суммирование операндов с плавающим знаком двойной точности (AVX512-512 бит)
extern __attribute__((target("avx512f,avx512bw"))) inline void dot_product_optimized_AVX512(double* a_ptr, double* w_ptr, size_t n, double& sum_val);
// Умножение и суммирование операндов с плавающим знаком одинарной точности (AVX512-512 бит)
extern __attribute__((target("avx512f,avx512bw"))) inline void dot_product_optimized_AVX512(float* a_ptr, float* w_ptr, size_t n, float& sum_val);
// Умножение и суммирование целочисленных 32-разрядных операндов (AVX512-512 бит)
extern __attribute__((target("avx512f,avx512bw"))) inline void dot_product_optimized_AVX512(int32_t* a_ptr, int32_t* w_ptr, size_t n, int32_t& sum_val);
// Умножение и суммирование целочисленных 8-разрядных операндов (AVX-256 бит)
__attribute__((target("avx2,fma"))) inline void dot_product_optimized_AVX256(char* a_ptr, char* w_ptr, size_t n, char& sum_val);
// Умножение и суммирование операндов с плавающим знаком двойной точности (AVX-256 бит)
__attribute__((target("avx2,fma"))) inline void dot_product_optimized_AVX256(double* a_ptr, double* w_ptr, size_t n, double& sum_val);
// Умножение и суммирование операндов с плавающим знаком одинарной точности (AVX-256 бит)
__attribute__((target("avx2,fma"))) inline void dot_product_optimized_AVX256(float* a_ptr, float* w_ptr, size_t n, float& sum_val);
// Умножение и суммирование целочисленных 32-разрядных операндов (AVX-256 бит)
__attribute__((target("avx2,fma"))) inline void dot_product_optimized_AVX256(int32_t* a_ptr, int32_t* w_ptr, size_t n, int32_t& sum_val);
// Умножение и суммирование целочисленных 8-разрядных операндов(SSE-128 бит)
__attribute__((target("sse4.2"))) inline void dot_product_optimized_SSE128(char* a_ptr, char* w_ptr, size_t n, char& sum_val);
// Умножение и суммирование операндов с плавающим знаком двойной точности (SSE-128 бит)
__attribute__((target("sse4.2"))) inline void dot_product_optimized_SSE128(double* a_ptr, double* w_ptr, size_t n, double& sum_val);
// Умножение и суммирование операндов с плавающим знаком одинарной точности (SSE-128 бит)
__attribute__((target("sse4.2"))) inline void dot_product_optimized_SSE128(float* a_ptr, float* w_ptr, size_t n, float& sum_val);
// Умножение и суммирование целочисленных 32-разрядных операндов(SSE-128 бит)
__attribute__((target("sse4.2"))) inline void dot_product_optimized_SSE128(int32_t* a_ptr, int32_t* w_ptr, size_t n, int32_t& sum_val);

// Класс умножения векторов и последующего суммирования
template <typename T>
class CVectorMulSum {
public:
    // Конструктор
    CVectorMulSum(const std::string& file1, const std::string& file2) {
        readData(file1, data1_);
        readData(file2, data2_);

        if (data1_.size() != data2_.size()) {
            throw std::runtime_error("Ошибка соответствия размеров файлов");
        }
    }

    // Умножение и суммирование с выбором векторных инструкций процессора
    T multiplyAndSum()
    {
        T sum = 0;
        size_t n = data1_.size()/sizeof(T);
        auto p1 = data1_.data();
        auto p2 = data1_.data();
        if ((int64_t)p1 % 64)
            p1 += 64 - ((int64_t)p1 % 64);
        if ((int64_t)p2 % 64)
            p2 += 64 - ((int64_t)p2 % 64);

        if (CheckOptimizationCPU(OPT_TYPE::AVX512))
            dot_product_optimized_AVX512((T *)p1, (T *)p2, n, sum);
        else
        if (CheckOptimizationCPU(OPT_TYPE::AVX))
            dot_product_optimized_AVX256((T *)p1, (T *)p2, n, sum);
        else
        if (CheckOptimizationCPU(OPT_TYPE::SSE))
            dot_product_optimized_SSE128((T *)p1, (T *)p2, n, sum);
        else
        {
            for (size_t i = 0; i < n; i++)
                sum += p1[i] * p2[i];
        }
        return sum;
    }
private:
    void readData(const std::string& file, std::vector<char>& data)
    {
        std::ifstream in(file, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Ошибка открытия файла " + file);
        }
        // Перемещение указателя чтения на конец файла
        in.seekg(0, std::ios::end);

        // Получение текущей позиции указателя чтения (количество байтов в файле)
        std::streamsize size = in.tellg();
        in.seekg(0, std::ios::beg);
        if (size > 1 << 20)
            throw std::runtime_error("Слишком большой размер файла");

        data.resize(size + 64);
        char *p = data.data();
        if ((int64_t)p % 64)
            p += 64 - ((int64_t)p % 64);
        T value;
        if (!in.read(p, size))
            throw std::runtime_error("Ошибка чтения файла");
    }

    std::vector<char> data1_;
    std::vector<char> data2_;
};

#endif // MULSUM_H
