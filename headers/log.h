#ifndef LOG_H
#define LOG_H

#include "utils.h"

// Уровни логирования
enum class LogLevel {
    kQuiet,     // Отключено
    kNormal,    // Норма
    kDebug,     // Избыточное для Debug
    kResult     // Пишется всегда (результаты)
};

// Класс записи в лог-файл/консоль
class Log
{
public:
    Log();

    // Задание имени лог-файла и режима логирования
    void SetLogName(const std::string &logname, bool quiet);

    // Установка уровня логирования
    void SetLogLevel(LogLevel level);

    // Запись сообщения в лог
    void WriteToLog(const std::string &str, LogLevel level = LogLevel::kNormal);

    // Проверка на наличие установленного лог-файла для записи
    bool IsLogFile() {return logname_.size() ?  !error_ : false;};
protected:
    // Ошибка при записи - запись в лог-файл отключена
    bool error_;

    // Уровень логирования для записи
    LogLevel log_level_;

    // Имя лог-файла, если отсутствует - то вывод идёт в консоль
    std::string logname_;
};

// Проверка на наличие установленного лог-файла для записи
bool IsLog();

// Установка уровня логирования
void SetLogLevel(LogLevel level);

// Запись в лог
void WriteToLog(const std::string &str, LogLevel level);

// Задание имени лог-файла и режима логирования
void SetLogName(const std::string &logname, bool quiet);

// Запись Debug-сообщений
#define LOG_DEBUG(str) WriteToLog(str, LogLevel::kDebug)
// Запись обычных-сообщений
#define LOG_INFO(str) WriteToLog(str, LogLevel::kNormal)
// Запись результатов
#define LOG_RESULT(str) WriteToLog(str, LogLevel::kResult)

#endif // LOG_H
