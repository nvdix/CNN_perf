#include "simpleconv.h"
#include "log.h"
#include "utils.h"

SimpleConv::SimpleConv(CONV_TYPE type, int tensor_height, int tensor_width, int height, int width, int inputs,
                       int outputs, int stride_vert, int stride_horiz, bool padding_vert, bool padding_horiz)
{
    conv_type_ = type;
    tensor_height_ = tensor_height;
    tensor_width_ = tensor_width;
    kernel_height_ = height;
    kernel_width_ = width;
    inputs_ = inputs;
    outputs_ = outputs;
    stride_vert_ = stride_vert;
    stride_horiz_ = stride_horiz;
    padding_vert_ = padding_vert;
    padding_horiz_ = padding_horiz;

    tensor_.resize(GetTensorSize());
    kernel_.resize(GetKernelSize() * inputs_ * outputs_);
    verify_outs_ = false;
}

SimpleConv::SimpleConv(ConvData *conv)
{
    *((ConvData *)this) = *conv;
    verify_outs_ = false;

    tensor_.resize(GetTensorSize());
    kernel_.resize(GetKernelSize() * inputs_ * outputs_);
}

float SimpleConv::RunConv()
{
    // Проверить размерности свёртки
    if (tensor_height_ < std::max(kernel_height_, stride_vert_) || tensor_width_ < std::max(kernel_width_, stride_horiz_))
        throw std::runtime_error(kErrorConvParam);

    std::string str;

    if (!accuracycheck_)
    {
        str = string_format("Расчёт свёртки (без оптимизации): %s", note_.data());
        LOG_INFO(str);

        str = string_format("Тензор-%dх%d; ядро-%dх%d; сдвиг-%dх%d; дополнение-%dх%d; входных каналов-%d; выходов-%d; тип-%s",
                        tensor_height_, tensor_width_, kernel_height_, kernel_width_, stride_horiz_, stride_vert_, padding_horiz_ ? 1 : 0,
                        padding_vert_ ? 1 : 0, inputs_, outputs_, GetTypeinString());
        LOG_INFO(str);
    }

    float result = 0;
    switch (conv_type_) {
    case CONV_TYPE::CONV_INT32:
        result = Convolution<int>();
        break;
    case CONV_TYPE::CONV_INT8:
        result = Convolution<char>();
        break;
    case CONV_TYPE::CONV_DOUBLE:
        result = Convolution<double>();
        break;
    case CONV_TYPE::CONV_FLOAT:
    default:
        result = Convolution<float>();
    }
    if (accuracycheck_ && !quietcheck_)
    {
        if (result <= gEpsilon)
            str = string_format("Тензор=%dх%d; ядро=%dх%d, сдвиг=%d,"
                  " дополнение -%s, тип=%3s - Успешно!",
                  tensor_height_, tensor_width_, kernel_height_,
                  kernel_width_, stride_horiz_, padding_horiz_ ?
                             " да" : "нет", GetTypeinString());
        else
            str = string_format("Тензор=%dх%d, ядро=%dх%d, сдвиг=%d,"
                  " дополнение -%3s, тип=%3s - Ошибка, расхождение = %.3f!",
                  tensor_height_, tensor_width_, kernel_height_,
                  kernel_width_, stride_horiz_, padding_horiz_ ?
                             " да" : "нет", GetTypeinString(), result);
        LOG_INFO(str);
    }
    return result;
}


// Выполнение многоканальной свёртки с расчётом времени
template<typename T>
float SimpleConv::Convolution()
{
    // Одноканальный тензор с добавлением padding если требуется
    std::vector<T> pad_tensor;
    FillPaddingTensor(pad_tensor);

    // Размеры выходной матрицы с учётом padding
    int ResHeight =1 + (tensor_height_ - (padding_horiz_ ? 1 : kernel_height_)) / stride_vert_;
    int ResWidth = 1 + (tensor_width_ - (padding_horiz_ ? 1 : kernel_width_)) / stride_horiz_;

    // Выходные матрицы
    std::vector<std::vector<T> > pad_results(outputs_);

    // Выделить память для новых тензоров
    for (int i = 0; i < pad_results.size(); ++i)
        pad_results[i].resize(ResHeight * ResWidth);

    int steps = 0;
    float timer = 0;

    FixTime();
    do
    {
        // Один проход рассчёта свёртки для каждого выхода
        for (int o = 0; o < outputs_; ++o)
        {
            // Заполнить результирующий тензор нулями
            std::memset(pad_results[o].data(), 0, pad_results[o].size() * sizeof(T));

            for (int i = 0; i < inputs_; ++i)
                MakeConv<T>(pad_results[o].data(), pad_tensor.data() + i * tensor_width_ * tensor_height_, ((T *)kernel_.data()) + (o * inputs_ + i) * GetKernelSize(false));
        }
        // Если идёт проверка результатов, то требуется только одна операция расчёта
        if (accuracycheck_)
            break;

        // Продолжаем пока время расчётов не превысит 1 секунду
        timer = GetTime();
        ++steps;
    } while (timer < 1.0);

    // Сверка результатов с эталоном - если требуются
    if (accuracycheck_)
    {
        // Начальные элементы массивов результата свёртки и эталонного значения
        T *newres = (T *) pad_results[0].data();
        T *fixres = (T *) result_.data();

        // Максимальная ошибка свёртки
        float maxerror = 0;

        for (int i = 0; i < GetResultSize(false); ++i)
        {
            float curerror = 0;
            // Вычисление ошибки взависимости от значения текщего элемента результата и эталонного значения

            if (newres[i] && fixres[i])
                curerror = std::abs(((float)fixres[i] - (float)newres[i])/std::max((float)std::abs(fixres[i]), (float)std::abs(newres[i])));

            // Сохранение текущей ошибки при её превышении над предыдущей
            if (maxerror < curerror)
                maxerror = curerror;
        }
        return maxerror;
    }
    return ((float)steps / timer);
}

// Свёртка одного канала
template<typename T>
int SimpleConv::MakeConv(T *result, T *tensor, T *kernel, dummyidentity<T>)
{
    // Размеры выходной матрицы с учётом дополнения
    int ResHeight =1 + (tensor_height_ - (padding_horiz_ ? 1 : kernel_height_)) / stride_vert_;
    int ResWidth = 1 + (tensor_width_ - (padding_horiz_ ? 1 : kernel_width_)) / stride_horiz_;

    // Размеры тензора с учётом дополнения
    int pad_tensor_width = tensor_width_ + (padding_horiz_ ? (kernel_width_ - 1) : 0);

    T* cursum = result;
    for (int y = 0; y < ResHeight; ++y)
    {
        // Верхний ряд в тензоре, участвующий в свёртке
        T *tensor_rowup = &tensor[y * stride_vert_ * pad_tensor_width];
        for (int x = 0; x < ResWidth; ++x)
        {
            // Верхний ряд в тензоре, смещённый относительно положения вычисляемого элемента
            T *tensor_rowcol = &tensor_rowup[x * stride_horiz_];
            for (int j = 0; j < kernel_height_; ++j)
            {
                // Текущий ряд ядра свёртки, смещённый относительно положения вычисляемого элемента
                T *kernel_currow = &kernel[j * kernel_width_];
                // Текущий элемент тензора совпадающий с началом текущего ряда свёртки
                T *tensor_cur = &tensor_rowcol[j * pad_tensor_width];

                // Суммирование произведений матрицы ядра и подматрицы тензора
                for (int i = 0; i < kernel_width_; ++i)
                    *cursum += kernel_currow[i] * tensor_cur[i];
            }

            // Инкремент указателя на текущий элемент результирующей матрицы
            ++cursum;
        }
    }

    return 0;
}

int SimpleConv::MakeConv(char *result, char *tensor, char *kernel, dummyidentity<char>)
{
    // Размеры выходной матрицы с учётом дополнения
    int ResHeight =1 + (tensor_height_ - (padding_horiz_ ? 1 : kernel_height_)) / stride_vert_;
    int ResWidth = 1 + (tensor_width_ - (padding_horiz_ ? 1 : kernel_width_)) / stride_horiz_;

    // Размеры тензора с учётом дополнения
    int pad_tensor_width = tensor_width_ + (padding_horiz_ ? (kernel_width_ - 1) : 0);

    char* cursum = result;
    int16_t sum = 0;
    for (int y = 0; y < ResHeight; ++y)
    {
        // Верхний ряд в тензоре, участвующий в свёртке
        char *tensor_rowup = &tensor[y * stride_vert_ * pad_tensor_width];
        for (int x = 0; x < ResWidth; ++x)
        {
            // Верхний ряд в тензоре, смещённый относительно положения вычисляемого элемента
            char *tensor_rowcol = &tensor_rowup[x * stride_horiz_];
            for (int j = 0; j < kernel_height_; ++j)
            {
                // Текущий ряд ядра свёртки, смещённый относительно положения вычисляемого элемента
                char *kernel_currow = &kernel[j * kernel_width_];
                // Текущий элемент тензора совпадающий с началом текущего ряда свёртки
                unsigned char *tensor_cur = (unsigned char *)&tensor_rowcol[j * pad_tensor_width];

                // Суммирование произведений матрицы ядра и подматрицы тензора
                for (int i = 0; i < kernel_width_; ++i)
                    sum += kernel_currow[i] * tensor_cur[i];
            }

            *cursum += sum >> 8;
            sum = 0;
            ++cursum;
        }
    }

    return 0;
}
