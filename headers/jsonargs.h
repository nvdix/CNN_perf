#ifndef JSONARGS_H
#define JSONARGS_H

#include "args.h"

// Класс работы с Json-файлами
class JsonArgs : public BaseArgs
{
public:
    // Конструктор с параметром - имя json-файла
    JsonArgs(const std::string &filename);

    // Построение списка свёрток
    int BuildNewConv() {return 0;};
};

#endif // JSONARGS_H
