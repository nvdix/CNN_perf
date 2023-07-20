#include <iostream>

#include "args.h"
#include "jsonargs.h"
#include "baseconv.h"
#include "log.h"

float gEpsilon = kDefEps;

int main(int argc, char *argv[])
{
    //SetLogLevel(LogLevel::kDebug);

    try {
        // Инициализация программного компонента разбора параметров командной строки
        std::shared_ptr<BaseArgs> arg = std::make_shared<CmdArgs>(argc, argv);
        std::string str;

        // Проверка на ключ затребующий вывод справки
        if (((CmdArgs *)arg.get())->FindHelp())
        {
            std::cout << kHelpMessage << std::endl;
            return 0;
        }

        // Проверить на наличие Json-конфигурации
        std::string config, logname;
        if (((CmdArgs *)arg.get())->FindJsonConfig(config))
            arg = std::make_shared<JsonArgs>(config);

        // Проверить необходимость логирования и тихого режима, установить их для лога
        bool quiet = arg->CheckLog(logname);
        SetLogName(logname, quiet);

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

        // Тесты свёрток
        if (arg->IsCorrectTests())
        {
            LOG_INFO("Тесты корректности свёрток:");
            TestArgs test;
            test.BuildTestConv();
            for (auto &&c : test.conv_vec_)
                std::to_string(c->RunConv());
        }

        // Создать список из заданных свёрток
        arg->BuildNewConv();

        str = string_format("Расчёт %d свёрток:", arg->conv_vec_.size());
        LOG_INFO(str);

        // Расчёт свёрток и вывод результатов
        for (auto &&c : arg->conv_vec_)
            LOG_RESULT(std::to_string(c->RunConv()));

    }
    // Обработчик исключений - вывод сообщений об ошибках
    catch(const std::exception& err)
    {
        std::cout << kErrorString << err.what() << std::endl;
        if (IsLog())
            LOG_RESULT(std::string(kErrorString) + err.what() + "\n");
        return 1;
    }

    return 0;
}
