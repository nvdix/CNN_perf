#ifndef SIMPLECONV_H
#define SIMPLECONV_H

#include "baseconv.h"

// Класс без оптимизации расчёта свёртки
class SimpleConv : public BaseConv
{
public:
    SimpleConv(ConvData *conv);

    SimpleConv(CONV_TYPE type, int tensor_height, int tensor_width, int height, int width, int inputs = 1,
               int outputs = 1, int stride_vert = 1, int stride_horiz = 1, bool padding_vert = false, bool padding_horiz = false);

    // Расчёт свёртки
    // Возвращает число рассчитанных за секунду свёрток
    float RunConv();

private:
    // Выполнение свёртки с расчётом времени
    // cmpdata [in] - выходные данные для проверки результатов
    template<typename T>
    float Convolution();

    // Расчёт свёртки
    template<typename T>
    int MakeConv(T *res, T *tensor, T *kernel) {return MakeConv(res, tensor, kernel, dummyidentity<T>());};

    // Расчёт свёртки для всех типов кроме int8
    template<typename T>
    int MakeConv(T *res, T *tensor, T *kernel, dummyidentity<T>);

    // Расчёт свёртки для int8
    int MakeConv(char *res, char *tensor, char *kernel, dummyidentity<char>);

    // Временный вектор для хранения данных умножения матриц
    std::vector<char> temptensor_;
};

#endif // SIMPLECONV_H
