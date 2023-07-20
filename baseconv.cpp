#include "simpleconv.h"
#include "sseconv.h"
#include "avxconv.h"
#include "avx512conv.h"
#include "log.h"
#include "utils.h"

ConvData::ConvData() :
    conv_type_(kDEF_TYPE),
    tensor_height_(kDEF_THEIGHT),
    tensor_width_(kDEF_TWIDTH),
    kernel_height_(kDEF_KHEIGHT),
    kernel_width_(kDEF_KWIDTH),
    inputs_(kDEF_INPUTS),
    outputs_(kDEF_OUTPUTS),
    stride_vert_(kDEF_VSTRIDE),
    stride_horiz_(kDEF_HSTRIDE),
    padding_vert_(kDEF_VPADDING),
    padding_horiz_(kDEF_HPADDING),
    verify_outs_(false)
{
}

int ConvData::GetTensorSize(bool inbytes)
{
    int size = tensor_width_ * tensor_height_ * inputs_;

    return inbytes ? size * GetTypeSize() : size;
}

int ConvData::GetKernelSize(bool inbytes)
{
    int size = kernel_width_ * kernel_height_;

    return inbytes ? size * GetTypeSize() : size;
}

int ConvData::GetResultSize(bool inbytes)
{
    // Размеры выходной матрицы с учётом padding
    int ResHeight =1 + (tensor_height_ - (padding_horiz_ ? 1 : kernel_height_)) / stride_vert_;
    int ResWidth = 1 + (tensor_width_ - (padding_horiz_ ? 1 : kernel_width_)) / stride_horiz_;
    int size = ResHeight * ResWidth;

    return inbytes ? size * GetTypeSize() : size;
}

int ConvData::GetTypeSize()
{
    int size;
    switch (conv_type_) {
    case CONV_TYPE::CONV_FLOAT:
        size = sizeof(float);
        break;
    case CONV_TYPE::CONV_INT32:
        size = sizeof(int);
        break;
    case CONV_TYPE::CONV_INT8:
        size = sizeof(char);
        break;
    case CONV_TYPE::CONV_DOUBLE:
        size = sizeof(double);
        break;
    default:
        throw std::runtime_error(kInternalError);
    }

    return size;
}

const char * ConvData::GetTypeinString()
{
    for (const auto& m : kDEF_Types)
        if (m.second == conv_type_)
            return m.first;

    throw std::runtime_error(kInternalError);
}

std::shared_ptr<BaseConv> BaseConv::CreateConv(OPT_TYPE opt_type, CONV_TYPE type, int tensor_height, int tensor_width, int height, int width,
                                               int inputs, int outputs, int stride_vert, int stride_horiz, bool padding_vert, bool padding_horiz)
{
    std::shared_ptr<BaseConv> newConv;

    switch (opt_type) {
    case OPT_TYPE::NONE:
        newConv = std::make_shared<SimpleConv>(type, tensor_height, tensor_width, height, width, inputs, outputs, stride_vert, stride_horiz, padding_vert, padding_horiz);
        break;
    case OPT_TYPE::SSE:
        newConv = std::make_shared<SSEConv>(type, tensor_height, tensor_width, height, width, inputs, outputs, stride_vert, stride_horiz, padding_vert, padding_horiz);
        break;
    case OPT_TYPE::AVX:
        newConv = std::make_shared<AVXConv>(type, tensor_height, tensor_width, height, width, inputs, outputs, stride_vert, stride_horiz, padding_vert, padding_horiz);
        break;
    case OPT_TYPE::AVX512:
        newConv = std::make_shared<AVX512Conv>(type, tensor_height, tensor_width, height, width, inputs, outputs, stride_vert, stride_horiz, padding_vert, padding_horiz);
        break;
    }

    return newConv;
}

std::shared_ptr<BaseConv> BaseConv::CreateConv(OPT_TYPE opt_type, ConvData *conv)
{
    std::shared_ptr<BaseConv> newConv;

    switch (opt_type) {
    case OPT_TYPE::NONE:
        newConv = std::make_shared<SimpleConv>(conv);
        break;
    case OPT_TYPE::SSE:
        newConv = std::make_shared<SSEConv>(conv);
        break;
    case OPT_TYPE::AVX:
        newConv = std::make_shared<AVXConv>(conv);
        break;
    case OPT_TYPE::AVX512:
        newConv = std::make_shared<AVX512Conv>(conv);
        break;
    }

    return newConv;
}

BaseConv::BaseConv() :
    accuracycheck_(false)
{

}
// Задать данные
int BaseConv::FillData(const char *tensor, int tensor_size, const char *kernel, int kernel_size)
{
    if (tensor && tensor_size && kernel && kernel_size)
    {
        // Проверка на соответствие размера данных
        if (tensor_size < GetTensorSize(true) ||
            kernel_size < GetKernelSize(true) * inputs_ * outputs_)
            throw std::runtime_error(kInternalError);

        std::memcpy(tensor_.data(), tensor, GetTensorSize(true));
        std::memcpy(kernel_.data(), kernel, GetKernelSize(true) * inputs_ * outputs_);
        return 0;
    }

    // Заполнение случайными данными
    {
        Rand rnd;

        // Заполнение входного тензора и ядра данными
        switch (conv_type_) {
        case CONV_TYPE::CONV_FLOAT:
            rnd.FillFloat((float *)tensor_.data(), GetTensorSize(false));
            rnd.FillFloat((float *)kernel_.data(), GetKernelSize(false) * inputs_ * outputs_);
            break;
        case CONV_TYPE::CONV_INT32:
            rnd.FillInt32((int *)tensor_.data(), GetTensorSize(false));
            rnd.FillInt32((int *)kernel_.data(), GetKernelSize(false) * inputs_ * outputs_);
            break;
        case CONV_TYPE::CONV_INT8:
            rnd.FillInt8(tensor_.data(), GetTensorSize(false));
            rnd.FillInt8(kernel_.data(), GetKernelSize(false) * inputs_ * outputs_, true);
            break;
        case CONV_TYPE::CONV_DOUBLE:
            rnd.FillDouble((double *)tensor_.data(), GetTensorSize(false));
            rnd.FillDouble((double *)kernel_.data(), GetKernelSize(false) * inputs_ * outputs_);
            break;
        }
    }

    return 0;
}

void BaseConv::SetResult(const char *data, int size, bool quiet)
{
    quietcheck_ = quiet;
    accuracycheck_ = true;
    result_.resize(size);
    std::memcpy(result_.data(), data, size);
//    std::copy(data, data + size, back_inserter(result_));
}

// Зафиксировать начальное время таймера расчёта задержки
void BaseConv::FixTime()
{
    timestart_ = std::chrono::high_resolution_clock::now();
}

// Получить время от старта таймера в секундах
float BaseConv::GetTime()
{
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - timestart_);

    return time_span.count();
}
