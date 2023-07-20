#ifndef CONSTS_H
#define CONSTS_H

#include <map>
#include <utility>
#include <iostream>
#include <cstring>
#include <thread>
#include <mutex>
#include <memory>
#include <stdio.h>

extern float gEpsilon;

typedef __attribute__(( aligned(64)))  char aligned_char;

// Типы данных в свёртке
enum class CONV_TYPE {
    CONV_FLOAT,
    CONV_DOUBLE,
    CONV_INT8,
    CONV_INT32
};

inline const std::map<const char* , CONV_TYPE> kDEF_Types = {
    {"f32", CONV_TYPE::CONV_FLOAT},
    {"f64", CONV_TYPE::CONV_DOUBLE},
    {"i8",  CONV_TYPE::CONV_INT8},
    {"i32", CONV_TYPE::CONV_INT32}
};

// Типы оптимизаций
enum class OPT_TYPE {
    NONE,
    SSE,
    AVX,
    AVX512
};

inline const std::map<const char* , OPT_TYPE> kDEF_Opts = {
    {"0", OPT_TYPE::NONE},
    {"N", OPT_TYPE::NONE},
    {"1", OPT_TYPE::SSE},
    {"S", OPT_TYPE::SSE},
    {"2", OPT_TYPE::AVX},
    {"A", OPT_TYPE::AVX},
    {"5", OPT_TYPE::AVX512}
};

// Наборы тестовых свёрток для верификации данных

// Tensor 5*5, Kernel 3*3, Stride = 1, No padding + Result
const double kTestConv0[] = {//Tensor
                            3,3,2,1,0,
                            0,0,1,3,1,
                            3,1,2,2,3,
                            2,0,0,2,2,
                            2,0,0,0,1,
                            //Kernel
                            0,1,2,
                            2,2,0,
                            0,1,2,
                            // Result
                            12,12,17,
                            10,17,19,
                            9,6,14};

// Tensor 5*5, Kernel 3*3, Stride = 2, Padding + Result
const double kTestConv1[] = {// Tensor
                            3,3,2,1,0,
                            0,0,1,3,1,
                            3,1,2,2,3,
                            2,0,0,2,2,
                            2,0,0,0,1,
                            // Kernel
                            0,1,2,
                            2,2,0,
                            0,1,2,
                            // Result
                            6,17,3,
                            8,17,13,
                            6,4,4};

// Tensor 5*5, Kernel 3*3, Stride = 1, No padding + Result
const float kTestConv2[] = {//Tensor
                            3,3,2,1,0,
                            0,0,1,3,1,
                            3,1,2,2,3,
                            2,0,0,2,2,
                            2,0,0,0,1,
                            //Kernel
                            0,1,2,
                            2,2,0,
                            0,1,2,
                            // Result
                            12,12,17,
                            10,17,19,
                            9,6,14};

// Tensor 5*5, Kernel 3*3, Stride = 2, Padding + Result
const float kTestConv3[] = {// Tensor
                            3,3,2,1,0,
                            0,0,1,3,1,
                            3,1,2,2,3,
                            2,0,0,2,2,
                            2,0,0,0,1,
                            // Kernel
                            0,1,2,
                            2,2,0,
                            0,1,2,
                            // Result
                            6,17,3,
                            8,17,13,
                            6,4,4};

// Tensor 5*5, Kernel 3*3, Stride = 1, No padding + Result
const int kTestConv4[] = {//Tensor
                          3,3,2,1,0,
                          0,0,1,3,1,
                          3,1,2,2,3,
                          2,0,0,2,2,
                          2,0,0,0,1,
                          //Kernel
                          0,1,2,
                          2,2,0,
                          0,1,2,
                          // Result
                          12,12,17,
                          10,17,19,
                          9,6,14};

// Tensor 5*5, Kernel 3*3, Stride = 2, Padding + Result
const int kTestConv5[] = {// Tensor
                          3,3,2,1,0,
                          0,0,1,3,1,
                          3,1,2,2,3,
                          2,0,0,2,2,
                          2,0,0,0,1,
                          // Kernel
                          0,1,2,
                          2,2,0,
                          0,1,2,
                          // Result
                          6,17,3,
                          8,17,13,
                          6,4,4};

// Tensor 5*5, Kernel 3*3, Stride = 1, No padding + Result
const char kTestConv6[] = {// Tensor
                          30,30,20,10,0,
                          0,0,10,30,10,
                          30,10,20,20,30,
                          20,0,0,20,20,
                          20,0,0,0,10,
                          // Kernel
                          0,10,20,
                          20,20,0,
                          0,10,20,
                          // Result
                          4,4,6,
                          3,6,7,
                          3,2,5};

// Tensor 5*5, Kernel 3*3, Stride = 2, Padding + Result
const char kTestConv7[] = {// Tensor
                          30,30,20,10,0,
                          0,0,10,30,10,
                          30,10,20,20,30,
                          20,0,0,20,20,
                          20,0,0,0,10,
                          // Kernel
                          0,10,20,
                          20,20,0,
                          0,10,20,
                          // Result
                          2,6,1,
                          3,6,5,
                          2,1,1};

// Предварительно заданные тестовые свёртки для проверки точности
inline const std::map<const char* , const void *> kDEF_TestConvs = {
    {"5,3,1,0,f64", kTestConv0},
    {"5,3,2,1,f64", kTestConv1},
    {"5,3,1,0,f32", kTestConv2},
    {"5,3,2,1,f32", kTestConv3},
    {"5,3,1,0,i32", kTestConv4},
    {"5,3,2,1,i32", kTestConv5},
    {"5,3,1,0,i8", kTestConv6},
    {"5,3,2,1,i8", kTestConv7}
};

// Максимальное число байт используемое при оптимизации (512бит / 8)
const int kMAX_OPT_BYTES = 64;

const int kDefMinTensor = 5;
const int kDefMaxTensor = 1024;

const int kDefMinKernel = 3;
const int kDefMaxKernel = 64;

const int kDefMinStride = 1;
const int kDefMaxStride = 32;

const int kDefMinInputs = 1;
const int kDefMaxInputs = 256;

const int kDefMinOutputs = 1;
const int kDefMaxOutputs = 256;

const OPT_TYPE kDEF_OPT = OPT_TYPE::NONE;
const CONV_TYPE kDEF_TYPE = CONV_TYPE::CONV_FLOAT;
const int kDEF_THEIGHT = 224;
const int kDEF_TWIDTH = 224;
const int kDEF_KHEIGHT = 3;
const int kDEF_KWIDTH = 3;
const int kDEF_INPUTS = 1;
const int kDEF_OUTPUTS = 1;
const int kDEF_HSTRIDE = 1;
const int kDEF_VSTRIDE = 1;
const bool kDEF_HPADDING = false;
const bool kDEF_VPADDING = false;

// Заданная точность Epsilon
const float kDefMinEps = 0.0000001;
const float kDefEps = 0.05;
const float kDefMaxEps = 0.9;

// Справка
const char * const kHelpMessage = "\
Использование: convbench [КЛЮЧ] [КЛЮЧ] …\n\
Выдаёт информацию о числе выполненных за секунду свёрток, заданных следующими параметрами:\n\
    -h, --help, -? - вызов данной справки с последующим выходом.\n\
    -q, -quiet — тихий режим. Выводится только результат расчёта свёртки.\n\
    -v, -verify — тестирование корректности выполнения свёрток.\n\
    -l=“log.txt“, -log=“log .txt“ - протоколирование вывода в указанный файл — log.txt. Имя файла может быть в кавычках (если содержит пробелы) или без оных. В этот файл дублируется вывод консоли.\n\
    -j=“config.json“, -config=“config.json“ - загрузка конфигурации из указанного файла — config.json. Имя файла может быть в кавычках (если содержит пробелы) или без оных. В этом файле задаются параметры работы утилиты.\n\
    -f=ZZZ, -factory=ZZZ - выбор предварительно заданных свёрток (одно или несколько значений). Свёртки задаются по номерам:\n\
        • 1 — LeNet-5 вход 32*32 ядро 5*5, сдвиг 1\n\
        • 2 — AlexNet вход 224*224 ядро 11*11, сдвиг 4, с дополнением\n\
        • 3 — VGG-19 вход 224*224 ядро 3*3, сдвиг 2, с дополнением\n\
        • 4 — ResNet вход 224*224 ядро 7*7, сдвиг 2, с дополнением\n\
        • 5 — SRCNN вход 32*32 ядро 9*9, сдвиг 1, с дополнением\n\
        Можно задавать несколько вариантов свёрток.\n\
        Например: «-f=145» означает: выбор 1,4,5 свёртки из списка с типом по умолчанию (если он не задан отдельным параметром).\n\
    -t=ZZZ, -type=ZZZ - выбор обсчитываемых типов данных. Может быть использовано несколько типов. Допустимые типы данных:\n\
        • f64 — double (знаковое с плавающей точкой 64 bit),\n\
        • f32 — float (знаковое с плавающей точкой 32 bit),\n\
        • i32 — int (знаковое целое 32 bit),\n\
        • i8 — char (знаковое целое 8 bit).\n\
        Если данный параметр присутствует, то он применяется ко всем свёрткам с не заданными типами данных. Если параметр нигде не указан, то для свёрток без указания типа выбирается значение по умолчанию f32.\n\
        Например: «-t=f32i32i8».\n\
        Данный параметр может присутствовать только в одном экземпляре.\n\
    -z=X, -optimize=X — оптимизация (X). Может быть:\n\
        • 0 — без оптимизации, стандарный алгоритм;\n\
        • 1 — SSE оптимизация (128 бит), усовершенствованный алгоритм;\n\
        • 2 — AVX оптимизация (256 бит), усовершенствованный алгоритм;\n\
        • 5 — AVX512 оптимизация (512 бит), усовершенствованный алгоритм.\n\
        Если данный параметр присутствует, то он применяется ко всем свёрткам с не заданной оптимизацией.\n\
        Может задаваться несколько вариантов оптимизации, например «-z0125».\n\
    -rH*W, -tensorH*W — размерность входного тензора. Где H — высота, W — ширина. H и W могут быть заданы в диапазоне от 16 до 512. По умолчанию используется H и W = 224.\n\
        Данный параметр может присутствовать только в одном экземпляре.\n\
    -kH*W -kernelH*W — размерность ядра. Где H — высота, W — ширина. H и W могут быть заданы в диапазоне от 3 до 31. По умолчанию используется H и W  = 3.\n\
        Данный параметр может присутствовать только в одном экземпляре.\n\
    -sH*V, -strideH*V — сдвиг (stride) по горизонтали и вертикали. Если указано одно число, то оно применяется на две оси. N — от 1 (по умолчанию) до 512. Сдвиг уменьшается автоматически по размеру тензора). \n\
        Данный параметр может присутствовать только в одном экземпляре.\n\
    -pH*V, -paddingH*V — дополнение нулями (padding). Может быть 0 (нет дополнения) или 1 (есть дополнение). Если указана одна цифра, то она применяется на две оси. По умолчанию сдвига нет.\n\
        Данный параметр может присутствовать только в одном экземпляре.\n\
    -iZ, -inputsZ — число входных каналов (Z). Может быть от 1 до 256.\n\
        Данный параметр может присутствовать только в одном экземпляре.\n\
    -oZ, -outputsZ — число выходов (Z). Может быть от 1 до 256.\n\
        Данный параметр может присутствовать только в одном экземпляре.\n\
    -cR,K,S,P,I,O,T,Z -convR,K,S,P,I,O,T,Z - выбор используемых свёрток (одно или несколько значений). Где:\n\
        • R - H*W — размерность входного тензора;\n\
        • K - H*W — размерность ядра;\n\
        • S - H*V — сдвиг (stride);\n\
        • P - H*V — дополнение нулями (padding);\n\
        • I — число входных каналов;\n\
        • O — число выходов;\n\
        • T — используемые типы данных;\n\
        • Z — оптимизация.\n\
        При этом обязательные параметры это размер тензора и ядра, остальные можно не указывать (будут использованы значения по умолчанию).\n\
        Данный параметр может присутствовать в нескольких экземплярах, т.е можно задать несколько различных свёрток. Например: «-c224*224,3*3,1*1,1 -c256,11,2,0,1,64,f32i8,0125»\n";

// Разделитель пар данных (высоты и ширины)
const char * const kHWDelimeter = "*";

// Разделитель между разными данными
const char *const kDataDelimeter = ",";

// Описание предопределённых свёрток
enum {
    kDEF_ID_LeNet = 1,
    kDEF_ID_AlexNet,
    kDEF_ID_VGG19,
    kDEF_ID_ResNet,
    kDEF_ID_SRCNN
};

inline const std::map<int, std::pair <const char*, const char* > > kDEF_CONVs = {
    {kDEF_ID_LeNet,     std::make_pair("32*32,5*5,1,0,1,6",       "LeNet-5")},
    {kDEF_ID_AlexNet,   std::make_pair("224*224,11*11,4,1,3,96",  "AlexNet")},
    {kDEF_ID_VGG19,     std::make_pair("224*224,3*3,2,1,3,64",    "VGG19")},
    {kDEF_ID_ResNet,    std::make_pair("224*224,7*7,2,1,3,64",    "ResNet")},
    {kDEF_ID_SRCNN,     std::make_pair("32*32,9*9,1,1,1,64",      "Super-Resolution Convolutional Neural Network")},
};

#endif // CONSTS_H
