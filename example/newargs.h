#ifndef NEWARGS_H
#define NEWARGS_H

#include <vector>
#include <string>
#include <memory>
#include <cstdlib>
#include <iostream>

#include "consts.h"

// Базовый класс разбора конфигурации
class BaseArgs
{
public:
    BaseArgs() {quietmode_ = false;};

    // Проверка необходимости логирования и тихого режима
    bool CheckLog() {return quietmode_;};

protected:
    // Признак тихого режима
    bool quietmode_;
};

// Класс описывающий получение данных из параметров при вызове утилиты
class CmdArgs : public BaseArgs
{
public:
    CmdArgs(int argc = 0, char *argv[] = nullptr);

    // Проверка на вызов справки
    bool FindHelp();

    // Получить имена входных файлов
    const std::string & GetFileIn1() {return infile1_;};
    const std::string & GetFileIn2() {return infile2_;};

    // Получить тип
    CONV_TYPE GetType() {return type_;};
protected:
    // Полный список всех заданных при вызове утилиты аргументов
    std::vector<std::string> args_;

    // Проверить параметр и выдать его дополнительную информацию
    bool CheckParam(std::string &str, const char *param1, const char *param2);

    // Имена входных файлов в бинарном виде
    std::string infile1_;
    std::string infile2_;

    // Установка типа данных
    bool SetType(const std::string &str);
    // Тип данных
    CONV_TYPE type_;
};

#endif // NEWARGS_H
