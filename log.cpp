#include <iostream>
#include <fstream>

#include "log.h"

Log gLog;

Log::Log()
    : log_level_ (LogLevel::kNormal),
      error_(false)
{

}

// Задание имени лог-файла и режима логирования
void Log::SetLogName(const std::string &logname, bool quiet)
{
    logname_ = logname;

    log_level_ = quiet ? LogLevel::kQuiet : LogLevel::kNormal;
}

// Установка уровня логирования
void Log::SetLogLevel(LogLevel level)
{
    log_level_ = level;
}

// Запись сообщения в лог
void Log::WriteToLog(const std::string &str, LogLevel level)
{
    if (logname_.size() && !error_)
    {
        std::ofstream out(logname_, std::ios::app);
        if (out.is_open())
        {
            if (log_level_ >= level  || level == LogLevel::kResult)
                out << str << std::endl;
            out.close();
            return;
        }
        else
        {
            if (!error_)
                std::cout << "Ошибка открытия файла логирования: "<< logname_ << " вывод осуществляется в консоль." << std::endl;
            error_ = true;
        }
    }
    if (log_level_ >= level  || level == LogLevel::kResult)
        std::cout << str << std::endl;
}

void WriteToLog(const std::string &str, LogLevel level)
{
    gLog.WriteToLog(str, level);
}

void SetLogName(const std::string &logname, bool quiet)
{
    gLog.SetLogName(logname, quiet);
}

void SetLogLevel(LogLevel level)
{
    gLog.SetLogLevel(level);
}

bool IsLog()
{
    return gLog.IsLogFile();
}
