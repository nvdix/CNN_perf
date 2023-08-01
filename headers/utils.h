#pragma once

#ifndef UTILS_H
#define UTILS_H

#include <stdarg.h>  // For va_start, etc.
#include <stdio.h>
#include <string.h>
#include <vector>
#include <memory>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <chrono>

#include "consts.h"

// Класс заполнения случайными данными указанных типов
class Rand
{
public:
    Rand() {
        std::srand(std::time(nullptr));
    }
    void FillFloat(float *buf, int elems) {
        for (int i = 0; i < elems; ++i)
            buf[i] = -1.0 + static_cast<float> (static_cast<double> (std::rand()) * 2 / RAND_MAX);
    };
    void FillDouble(double *buf, int elems) {
        for (int i = 0; i < elems; ++i)
            buf[i] = -1.0 + static_cast<double> (std::rand()) * 2 / RAND_MAX;
    };
    void FillInt8(char *buf, int elems, bool min = false) {
        for (int i = 0; i < elems; ++i)
            buf[i] = INT8_MIN/8 + static_cast<char> (static_cast<double>(std::rand()) * (INT8_MAX/8)/ RAND_MAX);
    };
    void FillInt32(int *buf, int elems) {
        for (int i = 0; i < elems; ++i)
            buf[i] = INT8_MIN + static_cast<int> (static_cast<double> (std::rand()) * INT8_MAX * 2 / RAND_MAX);
    };
};

// Форматирование в строку std::string с праметрами
std::string string_format(const std::string fmt_str, ...);

// Проверка наличия указанного вида оптимизации в процессоре
bool CheckOptimizationCPU(OPT_TYPE opt);

// Получение текстовой информации о процессоре
std::string GetCpuInfo();

#endif // UTILS_H
