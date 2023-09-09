#include <iostream>

#include "newargs.h"
#include "mulsum.h"
#include "log.h"

float gEpsilon = kDefEps;

// Справка
const char * const kHelpExampleMessage = "\
Использование: vecmulsm [КЛЮЧ] [КЛЮЧ] …\n\
Выдаёт сумму произведений векторов, заданных в бинарных файлах, заданных следующими параметрами:\n\
    -h, --help, -? - вызов данной справки с последующим выходом.\n\
    -q, -quiet — тихий режим. Выводится только результат расчёта.\n\
    -t=Z, -type=Z - выбор обсчитываемого типа данных. Допустимые типы данных:\n\
        • f64 — double (знаковое с плавающей точкой 64 bit),\n\
        • f32 — float (знаковое с плавающей точкой 32 bit),\n\
        • i32 — int (знаковое целое 32 bit),\n\
        • i8 — char (знаковое целое 8 bit).\n\
    -i=filename, -infile=filename — выбор входных файлов (filename). \n\
        Должны быть указаны два имени (в двух параметрах) бинарных файлов,\n\
        в каждом из которых сохранены данные вектора нужного типа.\n\
    Пример: vecmulsm -t=i8 -i=file1.bin -i=file2.bin ";

int main(int argc, char *argv[])
{
    //SetLogLevel(LogLevel::kDebug);

    try {
        // Инициализация программного компонента разбора параметров командной строки
        CmdArgs arg(argc, argv);
        std::string str;

        // Проверка на ключ затребующий вывод справки
        if (arg.FindHelp())
        {
            std::cout << kHelpExampleMessage << std::endl;
            return 0;
        }

        // Проверить необходимость логирования и тихого режима, установить их для лога
        bool quiet = arg.CheckLog();
        SetLogName("", quiet);


        std::vector<aligned_char> data(4);
        auto p = data.data();

        // Описание процессора
        if (!quiet)
        {
            str = "Процессор: ";
            str += GetCpuInfo();
            LOG_INFO(str);

            str = "Поддерживаемые расширения системы команд x86: ";
            str += CheckOptimizationCPU(OPT_TYPE::SSE) ? "SSE (128-бит)" : "нет.";
            str += CheckOptimizationCPU(OPT_TYPE::AVX) ? ", AVX (256-бит)" : "";
            str += CheckOptimizationCPU(OPT_TYPE::AVX512) ? ", AVX512 (512-бит)." : ".";
            LOG_INFO(str);
        }

        // Умножение векторов с суммированием и вывод результатов
        switch (arg.GetType()) {
        case CONV_TYPE::CONV_INT8:
            std::cout << (int)CVectorMulSum<char>(arg.GetFileIn1(), arg.GetFileIn2()).multiplyAndSum() << std::endl;
            break;
        case CONV_TYPE::CONV_INT32:
            std::cout << CVectorMulSum<int32_t>(arg.GetFileIn1(), arg.GetFileIn2()).multiplyAndSum() << std::endl;
            break;
        case CONV_TYPE::CONV_FLOAT:
            std::cout << CVectorMulSum<float>(arg.GetFileIn1(), arg.GetFileIn2()).multiplyAndSum() << std::endl;
            break;
        case CONV_TYPE::CONV_DOUBLE:
            std::cout << CVectorMulSum<double>(arg.GetFileIn1(), arg.GetFileIn2()).multiplyAndSum() << std::endl;
            break;
        }
    }
    // Обработчик исключений - вывод сообщений об ошибках
    catch(const std::exception& err)
    {
        std::cout << kErrorString << err.what() << std::endl;
        return 1;
    }

    return 0;
}
