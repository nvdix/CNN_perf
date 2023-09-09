#include <cstring>

#include "headers/args.h"

bool BaseArgs::SetType(const std::string &str, BaseArgs * conv)
{
    if (!conv)
        conv = this;

    std::vector<CONV_TYPE> types;

    for (const auto& m : kDEF_Types)
        if (str.find(m.first) != std::string::npos)
            types.push_back(m.second);

//    if (!types.size())
//        types.push_back(kDEF_TYPE);

    if (types.size())
        conv->conv_type_ = types[0];

    conv->types_ = types;
    return true;
}

bool BaseArgs::SetTensor(const std::string &str, BaseArgs * conv)
{
    if (!conv)
        conv = this;

    char *end;
    conv->tensor_height_ = strtoul(str.data(), &end, 10);
    auto i = str.find(kHWDelimeter);
    auto n = str.find(kDataDelimeter);

    // Проверка на расхождение разделителей
    if (n != std::string::npos && i != std::string::npos && n < i)
        i = std::string::npos;
    unsigned long temp = strtoul(&str[i + 1], &end, 10);

    conv->tensor_width_ = i != std::string::npos ? temp : conv->tensor_height_;

    if (std::min(conv->tensor_height_, conv->tensor_width_) < kDefMinTensor ||
        std::max(conv->tensor_height_, conv->tensor_width_) > kDefMaxTensor)
        throw std::runtime_error(kErrorTensorParam);

    return true;
}

bool BaseArgs::SetKernel(const std::string &str, BaseArgs * conv)
{
    if (!conv)
        conv = this;

    char *end;
    conv->kernel_height_ = strtoul(str.data(), &end, 10);
    auto i = str.find(kHWDelimeter);
    auto n = str.find(kDataDelimeter);

    // Проверка на расхождение разделителей
    if (n != std::string::npos && i != std::string::npos && n < i)
        i = std::string::npos;

    unsigned long temp = strtoul(&str[i + 1], &end, 10);

    conv->kernel_width_ = i != std::string::npos ? temp : conv->kernel_height_;

    if (std::min(conv->kernel_height_, conv->kernel_width_) < kDefMinKernel ||
        std::max(conv->kernel_height_, conv->kernel_width_) > kDefMaxKernel)
        throw std::runtime_error(kErrorKernelParam);

    return true;
}

bool BaseArgs::SetStride(const std::string &str, BaseArgs * conv)
{
    if (!conv)
        conv = this;

    char *end;
    conv->stride_vert_ = strtoul(str.data(), &end, 10);
    auto i = str.find(kHWDelimeter);
    auto n = str.find(kDataDelimeter);

    // Проверка на расхождение разделителей
    if (n != std::string::npos && i != std::string::npos && n < i)
        i = std::string::npos;

    unsigned long temp = strtoul(&str[i + 1], &end, 10);

    conv->stride_horiz_ = i != std::string::npos ? temp : conv->stride_vert_;
    if (std::min(conv->stride_horiz_, conv->stride_vert_) < kDefMinStride ||
        std::max(conv->stride_horiz_, conv->stride_vert_) > kDefMaxStride)
        throw std::runtime_error(kErrorStrideParam);

    return true;
}

bool BaseArgs::SetPadding(const std::string &str, BaseArgs * conv)
{
    if (!conv)
        conv = this;

    char *end;
    conv->padding_vert_ = strtoul(str.data(), &end, 10);
    auto i = str.find(kHWDelimeter);
    auto n = str.find(kDataDelimeter);

    // Проверка на расхождение разделителей
    if (n != std::string::npos && i != std::string::npos && n < i)
        i = std::string::npos;

    unsigned long temp = strtoul(&str[i + 1], &end, 10);

    conv->padding_horiz_ = i != std::string::npos ? temp : conv->padding_vert_;

    return true;
}

bool BaseArgs::SetInputs(const std::string &str, BaseArgs * conv)
{
    if (!conv)
        conv = this;

    char *end;
    conv->inputs_ = strtoul(str.data(), &end, 10);
    if (conv->inputs_ < kDefMinInputs ||
        conv->inputs_ > kDefMaxInputs)
        throw std::runtime_error(kErrorInputsParam);

    return true;
}

bool BaseArgs::SetOutputs(const std::string &str, BaseArgs * conv)
{
    if (!conv)
        conv = this;

    char *end;
    conv->outputs_ = strtoul(str.data(), &end, 10);

    if (conv->outputs_ < kDefMinOutputs ||
        conv->outputs_ > kDefMaxOutputs)
        throw std::runtime_error(kErrorOutputsParam);

    return true;
}

bool BaseArgs::SetOptimize(const std::string &str, BaseArgs * conv)
{
    if (!conv)
        conv = this;

    std::vector<OPT_TYPE> opts;

    for (const auto& m : kDEF_Opts)
        if (str.find(m.first) != std::string::npos && CheckOptimizationCPU(m.second))
            opts.push_back(m.second);

//    if (!opts.size())
//        opts.push_back(kDEF_OPT);

    conv->opts_ = opts;
    return true;
}

// Проверка на наличие следующих данных и удаление обработанных
// Возвращает true если есть следующий токен с данными
bool BaseArgs::RemoveFirstToken(std::string &conv)
{
    if (conv.find(kDataDelimeter) == std::string::npos)
        return false;
    conv.erase(0, conv.find(kDataDelimeter) + 1);
    return true;
}

CmdArgs::CmdArgs(int argc, char *argv[])
{
    if (!argc || !argv)
        return;

    args_.resize(argc);

    for (int i = 0; i < argc; ++i)
        args_[i] = argv[i];

    // Проверка на логирование и тихий режим
    for (auto a : args_)
    {
        if (CheckParam(a, "-q", "-quiet"))
            quietmode_ = true;
        else
        if (CheckParam(a, "-l=", "-log="))
            logname_ = a;
        else
        if (CheckParam(a, "-v=", "-verify=") || CheckParam(a, "-v", "-verify"))
        {
            if (a.size())
            {
                char *pend;
                float eps = strtof32(a.data(), &pend);
                if (eps >= kDefMinEps && eps <= kDefMaxEps)
                    gEpsilon = eps;
                else
                    throw std::runtime_error(kErrorEpsilon);
            }
            correcttests_ = true;
            verify_outs_ = true;
        }
    }
}

// Проверка справки
bool CmdArgs::FindHelp()
{
    for (auto&& a : args_)
        if (a == "--help" || a == "-h" || a == "-?")
            return true;
    return false;
}

// Проверка на указание файла конфигурации
bool CmdArgs::FindJsonConfig(std::string &filename)
{
    for (auto a : args_)
        if (CheckParam(a, "-j=", "-config="))
        {
            filename = a;
            return filename.size() ? true : false;
        }
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

// Проверить входные параметры и установить значения по умолчанию
bool CmdArgs::SetDefaults()
{
    bool result = true;
    std::string str;
    for (auto a : args_)
    {
        if (CheckParam(a, "-t=", "-type="))
            result = SetType(a);
        else
        if (CheckParam(a, "-r", "-tensor"))
            result = SetTensor(a);
        else
        if (CheckParam(a, "-i", "-inputs"))
            result = SetInputs(a);
        else
        if (CheckParam(a, "-o", "-outputs"))
            result = SetOutputs(a);
        else
        if (CheckParam(a, "-k", "-kernel"))
            result = SetKernel(a);
        else
        if (CheckParam(a, "-s", "-stride"))
            result = SetStride(a);
        else
        if (CheckParam(a, "-p", "-padding"))
            result = SetPadding(a);
        else
        if (CheckParam(a, "-z=", "-optimize="))
            result = SetOptimize(a);

        if (!result)
            return false;
    }

    return true;
}

// Добавить заранее заданные свёртки
void CmdArgs::AddPredefinedConv(const std::string &str)
{
    for (const auto& m : kDEF_CONVs)
        if (str.find(std::to_string(m.first)) != std::string::npos)
           AddConv(m.second.first, m.second.second);
}

// Добавление новой свёртки
// conv - [in] - строка с данными свёртки
void CmdArgs::AddConv(std::string conv, std::string note)
{
    BaseArgs newConv = *this;

    do {
        // Установить данные размера тензора
        if (!SetTensor(conv, &newConv))
            return;

        // Проверка на наличие следующих данных и удаление обработанных
        if (!RemoveFirstToken(conv))
            return;

        // Установить данные размера ядра
        if (!SetKernel(conv, &newConv))
            return;

        // Проверка на наличие следующих данных и удаление обработанных
        if (!RemoveFirstToken(conv))
            break;

        // Установить данные прореживания
        if (!SetStride(conv, &newConv))
            return;

        // Проверка на наличие следующих данных и удаление обработанных
        if (!RemoveFirstToken(conv))
            break;

        // Установить дополнение нулями
        if (!SetPadding(conv, &newConv))
            break;

        // Проверка на наличие следующих данных и удаление обработанных
        if (!RemoveFirstToken(conv))
            break;

        // Установить число входов
        if (!SetInputs(conv, &newConv))
            break;

        // Проверка на наличие следующих данных и удаление обработанных
        if (!RemoveFirstToken(conv))
            break;

        // Установить число выходов
        if (!SetOutputs(conv, &newConv))
            break;

        // Проверка на наличие следующих данных и удаление обработанных
        if (!RemoveFirstToken(conv))
            break;

        // Установить число выходов
        if (!SetType(conv, &newConv))
            break;

        // Проверка на наличие следующих данных и удаление обработанных
        if (!RemoveFirstToken(conv))
            break;

        // Установить число выходов
        if (!SetOptimize(conv, &newConv))
            break;

        // Проверка на наличие следующих данных и удаление обработанных
        if (!RemoveFirstToken(conv))
            break;
    } while (0);

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
        conv_vec_.back()->SetNote(note);

        // Задать случайные данные
        conv_vec_.back()->FillData();
    }
}

// Построение вектора с заданными параметрами свёртками
// Возвращает число свёрток в векторе
int CmdArgs::BuildNewConv()
{
    SetDefaults();

    // Проверка на предварительно заданные свёртки
    for (auto a : args_)
        if (CheckParam(a, "-f=", "-factory="))
        {
            AddPredefinedConv(a);
            break;
        }

    // Добавление заданных параметрами свёрток
    for (auto a : args_)
        if (CheckParam(a, "-c", "-conv"))
            AddConv(a);

    if (!conv_vec_.size())
        throw std::runtime_error(kErrorNoConvolutions);

    return conv_vec_.size();
}

// Построение вектора с тестовым набором свёрток
// Возвращает число свёрток в векторе
int TestArgs::BuildTestConv()
{
    for (const auto& m : kDEF_TestConvs)
        AddTestConv(m.first, (const char*)m.second);

    return conv_vec_.size();
}

// Добавление тестовой свёртки
// conv - [in] строка с данными свёртки
// data - [in] данные свёртки
void TestArgs::AddTestConv(std::string conv, const char * data)
{
    BaseArgs newConv = *this;

    do {
        // Установить данные размера тензора
        if (!SetTensor(conv, &newConv))
            return;

        // Проверка на наличие следующих данных и удаление обработанных
        if (!RemoveFirstToken(conv))
            return;

        // Установить данные размера ядра
        if (!SetKernel(conv, &newConv))
            return;

        // Проверка на наличие следующих данных и удаление обработанных
        if (!RemoveFirstToken(conv))
            break;

        // Установить данные прореживания
        if (!SetStride(conv, &newConv))
            return;

        // Проверка на наличие следующих данных и удаление обработанных
        if (!RemoveFirstToken(conv))
            break;

        // Установить дополнение нулями
        if (!SetPadding(conv, &newConv))
            break;

        // Проверка на наличие следующих данных и удаление обработанных
        if (!RemoveFirstToken(conv))
            break;

        // Установить тип
        if (!SetType(conv, &newConv))
            break;

    } while (0);

//    newConv.inputs_ = 3;

    // Создать и добавить все заданные типы свёрток в вектор
    conv_vec_.push_back(BaseConv::CreateConv(OPT_TYPE::NONE, &newConv));

//    std::vector<char> t(newConv.GetTensorSize());
//    std::vector<char> k(newConv.GetKernelSize() * 3);

//    newConv.inputs_ = 1;
//    float *buf = (float *)t.data();
//    for (int i = 0; i < 25; ++i)
//        buf[i*3] = buf[i*3 + 1] = buf[i*3 + 2] = ((float *)data)[i];

//    for (int i = 0; i < 3; ++i)
//        std::memcpy(k.data() + i * newConv.GetKernelSize(), data + newConv.GetTensorSize(), newConv.GetKernelSize());

//    // Задать данные
//    conv_vec_.back()->FillData(t.data(), newConv.GetTensorSize() * 3, k.data(), newConv.GetKernelSize() * 3);
//    newConv.inputs_ = 1;
//    conv_vec_.back()->SetResult(data + newConv.GetTensorSize() + newConv.GetKernelSize(), newConv.GetResultSize());

    // Заполнение данными
    conv_vec_.back()->FillData(data, newConv.GetTensorSize(), data + newConv.GetTensorSize(), newConv.GetKernelSize());
    conv_vec_.back()->SetResult(data + newConv.GetTensorSize() + newConv.GetKernelSize(), newConv.GetResultSize());
}
