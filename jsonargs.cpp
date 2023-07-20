#include <fstream>
#include <nlohmann/json.hpp>

#include "jsonargs.h"

JsonArgs::JsonArgs(const std::string &filename)
{
    std::string config;

    // Чтение конфигурационного Json файла
    std::ifstream in(filename);
    if (!in.is_open())
        throw std::runtime_error(kErrorConfigFile);

    config = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    // Парсинг Json
    nlohmann::json out_json = nlohmann::json::parse(config.data());

    // Данные конфигурации

    // Тихий режим
    if (out_json["config"]["quiet"].is_boolean())
        quietmode_ = out_json["config"]["quiet"].get<bool>();

    // Лог-файл
    if (out_json["config"]["log"].is_string())
        logname_ = out_json["config"]["log"].get<std::string>();

    // Тесты точности
    if (out_json["config"]["verify"].is_string() || out_json["config"]["verify"].is_boolean() || out_json["config"]["verify"].is_number_float())
    {
        if (out_json["config"]["verify"].is_boolean() && out_json["config"]["verify"].get<bool>())
            correcttests_ = true;
        else
        {
            correcttests_ = true;
            float eps = 2 * kDefMaxEps;
            if (out_json["config"]["verify"].is_string())
                eps = std::stof(out_json["config"]["verify"].get<std::string>());
            else
            if (out_json["config"]["verify"].is_number_float())
                eps = out_json["config"]["verify"].get<float>();

            if (eps >= kDefMinEps && eps <= kDefMaxEps)
                gEpsilon = eps;
            else
                throw std::runtime_error(kErrorEpsilon);
        }
    }

    if (!out_json["conv"].size())
        throw std::runtime_error(kErrorNoConvolutions);

    BaseArgs newConv;

    for (int i = 0; i < out_json["conv"].size(); ++i)
    {
        newConv = *this;
        auto curjson = out_json["conv"][i];

        //std::cout << out_json["conv"][i];

        if (!curjson["tensor"].is_string())
            throw std::runtime_error(kErrorNoTensor);

        if (!curjson["kernel"].is_string())
            throw std::runtime_error(kErrorNoKernel);

        // Установить данные размера тензора
        if (!SetTensor(curjson["tensor"].get<std::string>(), &newConv))
            return;

        // Установить данные размера ядра
        if (!SetKernel(curjson["kernel"].get<std::string>(), &newConv))
            return;

        // Установить данные прореживания
        if (curjson["stride"].is_string() && !SetStride(curjson["stride"].get<std::string>(), &newConv))
            return;

        // Установить дополнение нулями
        if (curjson["padding"].is_string() && !SetPadding(curjson["padding"].get<std::string>(), &newConv))
            return;

        // Установить число входов
        if (curjson["inputs"].is_string() && !SetInputs(curjson["inputs"].get<std::string>(), &newConv))
            return;

        // Установить число выходов
        if (curjson["outputs"].is_string() && !SetOutputs(curjson["outputs"].get<std::string>(), &newConv))
            return;

        // Установить типы
        if (curjson["types"].is_string())
        {
            std::string t = curjson["types"].get<std::string>();
            if (!SetType(t, &newConv))
                return;
        }

        // Установить оптимизацию
        if (curjson["optimisation"].is_string())
        {
            std::string o = curjson["optimisation"].get<std::string>();
            if (!SetOptimize(o, &newConv))
                return;
        }

        // Установить описание
        if (curjson["note"].is_string())
            newConv.note_ = curjson["note"].get<std::string>();

        // Создать и добавить все заданные типы свёрток в вектор
        if (!newConv.types_.size())
        {
            if (!types_.size())
                newConv.types_.push_back(conv_type_);
            else
                newConv.types_ = types_;
        }

        if (!newConv.opts_.size())
        {
            if (!opts_.size())
                newConv.opts_.push_back(kDEF_OPT);
            else
                newConv.opts_ = opts_;
        }

        for (auto o : newConv.opts_)
        for (auto t : newConv.types_)
        {
            newConv.conv_type_ = t;
            conv_vec_.push_back(BaseConv::CreateConv(o, &newConv));

            // Задать случайные данные
            conv_vec_.back()->FillData();
        }
    }
}
