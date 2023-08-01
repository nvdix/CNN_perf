#ifndef ERRORS_H
#define ERRORS_H

const char * const kErrorString = "Программа остановлена из-за ошибки: ";
const char * const kErrorTensorParam = "Задан неверный размер тензора.";
const char * const kErrorKernelParam = "Задан неверный размер ядра.";
const char * const kErrorStrideParam = "Задан неверный размер сдвига.";
const char * const kErrorPaddingParam = "Дополнение нулями задано с ошибкой.";
const char * const kErrorInputsParam = "Задано неверное число входных каналов";
const char * const kErrorOutputsParam = "Задано неверное число выходов";

const char * const kErrorNoTensor = "Ошибка чтения размера тензора.";
const char * const kErrorNoKernel = "Ошибка чтения размера ядра.";
const char * const kErrorEpsilon = "Ошибка в параметре верификации - неверное значение параметра epsilon.";
const char * const kErrorConvParam = "Размер тензора меньше размеров ядра или сдвига.";

const char * const kErrorNoConvolutions = "Отсутствуют свёртки для расчёта.";

const char * const kErrorConfigFile = "Ошибка открытия файла конфигурации.";

const char * const kInternalError = "Внутренняя ошибка выполнения.";

#endif // ERRORS_H
