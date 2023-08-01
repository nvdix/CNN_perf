#ifndef ARGS_H
#define ARGS_H

#include <vector>
#include <string>
#include <memory>
#include <cstdlib>
#include <iostream>

#include "baseconv.h"

// Базовый класс разбора конфигурации
class BaseArgs : public ConvData
{
public:
    BaseArgs() {quietmode_ = correcttests_ = false;};

    // Список типов для расчёта свёрток
    std::vector<CONV_TYPE> types_;

    // Список видов оптимизаций
    std::vector<OPT_TYPE> opts_;

    // Построение вектора с заданными параметрами свёртками
    // Возвращает число свёрток в векторе
    virtual int BuildNewConv() {throw std::runtime_error(kInternalError);};

    // Проверка необходимости логирования и тихого режима
    bool CheckLog(std::string &str) {
            str = logname_;
            return quietmode_;};

    // Вектор классов свёрток
    std::vector<std::shared_ptr<BaseConv> > conv_vec_;

    bool IsCorrectTests() {return correcttests_;};

protected:
    // Установка данных свёртки из строки
    bool SetType(const std::string &str, BaseArgs * conv = nullptr);
    bool SetTensor(const std::string &str, BaseArgs * conv = nullptr);
    bool SetKernel(const std::string &str, BaseArgs * conv = nullptr);
    bool SetStride(const std::string &str, BaseArgs * conv = nullptr);
    bool SetPadding(const std::string &str, BaseArgs * conv = nullptr);
    bool SetInputs(const std::string &str, BaseArgs * conv = nullptr);
    bool SetOutputs(const std::string &str, BaseArgs * conv = nullptr);
    bool SetOptimize(const std::string &str, BaseArgs * conv = nullptr);

    // Проверка на наличие следующих данных и удаление обработанных
    bool RemoveFirstToken(std::string &conv);

    // Признак тестов верификации расчёта свёрток
    bool correcttests_;

    // Признак тихого режима
    bool quietmode_;

    // Имя лог-файла, если есть
    std::string logname_;
};

// Класс описывающий получение данных из параметров при вызове утилиты
class CmdArgs : public BaseArgs
{
public:
    CmdArgs(int argc = 0, char *argv[] = nullptr);

    // Проверка на вызов справки
    bool FindHelp();

    // Проверка на указание файла конфигурации
    bool FindJsonConfig(std::string &filename);

    // Проверить входные параметры и установить значения по умолчанию
    bool SetDefaults();

    // Построение вектора с заданными параметрами свёртками
    // Возвращает число свёрток в векторе
    int BuildNewConv();

protected:
    // Полный список всех заданных при вызове утилиты аргументов
    std::vector<std::string> args_;

    // Проверить параметр и выдать его дополнительную информацию
    bool CheckParam(std::string &str, const char *param1, const char *param2);

    void AddPredefinedConv(const std::string &str);

    // Добавление новой свёртки
    // conv - [in] - строка с данными свёртки
    void AddConv(std::string str, std::string note = "");
};

// Класс описывающий верификацию предварительно заданных сборок
class TestArgs : public CmdArgs
{
public:
    // Добавление тестовой свёртки
    // conv - [in] строка с данными свёртки
    // data - [in] данные свёртки
    void AddTestConv(std::string conv, const char * data);

    // Построение вектора с заданными параметрами свёртками
    // Возвращает число свёрток в векторе
    int BuildTestConv();

protected:
    // Полный список всех заданных при вызове утилиты аргументов
    std::vector<std::string> args_;
};

#endif // ARGS_H
