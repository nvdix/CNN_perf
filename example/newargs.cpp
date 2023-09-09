#include <cstring>

#include "newargs.h"

CmdArgs::CmdArgs(int argc, char *argv[])
{
    if (!argc || !argv)
        return;

    args_.resize(argc);

    for (int i = 0; i < argc; ++i)
        args_[i] = argv[i];

    type_ = CONV_TYPE::CONV_INT32;
    // Проверка всех ключей
    for (auto a : args_)
    {
        if (CheckParam(a, "-q", "-quiet"))
            quietmode_ = true;
        else
        if (CheckParam(a, "-i=", "-infile="))
        {
            if (infile1_.size())
                infile2_ = a;
            else
                infile1_ = a;
        }
        else
        if (CheckParam(a, "-t=", "-type="))
            SetType(a);
    }
}

// Установка типа данных
bool CmdArgs::SetType(const std::string &str)
{
    for (const auto& m : kDEF_Types)
        if (str.find(m.first) != std::string::npos)
        {
            type_ = m.second;
            break;
        }

    return true;
}

// Проверка справки
bool CmdArgs::FindHelp()
{
    for (auto&& a : args_)
        if (a == "--help" || a == "-h" || a == "-?")
            return true;
    return false;
}

// Проверить параметр и выдать его дополнительную информацию
bool CmdArgs::CheckParam(std::string &str, const char *param1, const char *param2)
{
    if (str.find(param1) != std::string::npos)
    {
        str.erase(0, strlen(param1));
        return true;
    }

    if (str.find(param2) != std::string::npos)
    {
        str.erase(0, strlen(param2));
        return true;
    }

    return false;
}
