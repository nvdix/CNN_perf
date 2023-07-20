#include <mmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>

#include "simpleconv.h"
#include "sseconv.h"
#include "log.h"
#include "utils.h"

SSEConv::SSEConv(CONV_TYPE type, int tensor_height, int tensor_width, int height, int width, int inputs,
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
}

SSEConv::SSEConv(ConvData *conv)
{
    *((ConvData *)this) = *conv;

    tensor_.resize(GetTensorSize());
    kernel_.resize(GetKernelSize() * inputs_ * outputs_);
}

float SSEConv::RunConv()
{
    // Проверить размерности свёртки
    if (tensor_height_ < std::max(kernel_height_, stride_vert_) || tensor_width_ < std::max(kernel_width_, stride_horiz_))
        throw std::runtime_error(kErrorConvParam);

    std::string str;

    if (!accuracycheck_)
    {
        str = string_format("Расчёт свёртки (SSE 128 бит оптимизация): %s", note_.data());
        LOG_INFO(str);
    }

    if (!accuracycheck_ && !verify_outs_)
    {
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

    if (accuracycheck_)
    {
        if (result <= gEpsilon)
            str = string_format("Тензор=%dх%d; ядро=%dх%d, сдвиг=%d,"
                  " дополнение -%s, тип=%3s - Успешно!",
                  tensor_height_, tensor_width_, kernel_height_,
                  kernel_width_, stride_horiz_, padding_horiz_ ?
                             " да" : "нет", GetTypeinString());
        else
            str = string_format("Тензор=%dх%d, ядро=%dх%d, сдвиг=%d,"
                  " дополнение -%3s, тип=%3s - Ошибка, расхождение = %.2f%%!",
                  tensor_height_, tensor_width_, kernel_height_,
                  kernel_width_, stride_horiz_, padding_horiz_ ?
                             " да" : "нет", GetTypeinString(), result * 100);
        LOG_INFO(str);
    }
    else
    if (verify_outs_)
    {
        if (verify_result_ < gEpsilon)
            str = string_format("Тензор-%dх%d; ядро-%dх%d; сдвиг-%dх%d; дополнение-%dх%d; входных каналов-%d; выходов-%d; тип-%s - верификация пройдена успешно!",
                            tensor_height_, tensor_width_, kernel_height_, kernel_width_, stride_horiz_, stride_vert_, padding_horiz_ ? 1 : 0,
                            padding_vert_ ? 1 : 0, inputs_, outputs_, GetTypeinString());
        else
            str = string_format("Тензор-%dх%d; ядро-%dх%d; сдвиг-%dх%d; дополнение-%dх%d; входных каналов-%d; выходов-%d; тип-%s - ошибка верификации = %.2f%%!",
                            tensor_height_, tensor_width_, kernel_height_, kernel_width_, stride_horiz_, stride_vert_, padding_horiz_ ? 1 : 0,
                            padding_vert_ ? 1 : 0, inputs_, outputs_, GetTypeinString(), verify_result_ * 100);
        LOG_INFO(str);
    }

    return result;
}

// Умножение и суммирование целочисленных 8-разрядных операндов(SSE-128 бит)
__attribute__((target("sse4.2"))) inline void dot_product_optimized_SSE128(char* a_ptr, char* w_ptr, size_t n, char& sum_val)
{
    __m128i sum4 = _mm_setzero_si128();
    int32_t* sum_ptr = (int32_t*)&sum4;
    __m128i* a_ptr16 = (__m128i*)a_ptr;
    __m128i* w_ptr16 = (__m128i*)w_ptr;
    for (size_t i = 0; i < n; ++i)
    {
        __m128i c=_mm_maddubs_epi16(a_ptr16[i], w_ptr16[i]);
        __m128i lo=_mm_cvtepi16_epi32(c);
        __m128i hi=_mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum4 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum4);
    }
    int &s = *((int *)&sum_val);
    s = sum_ptr[0] + sum_ptr[1] + sum_ptr[2] + sum_ptr[3];
}

// Умножение и суммирование операндов с плавающим знаком двойной точности (SSE-128 бит)
__attribute__((target("sse4.2"))) inline void dot_product_optimized_SSE128(double* a_ptr, double* w_ptr,size_t n, double& sum_val)
{
    __m128d sum2 = _mm_setzero_pd();
    double* sum_ptr = (double *) &sum2;
    __m128d* a_ptr2 = (__m128d*)a_ptr;
    __m128d* w_ptr2 = (__m128d*)w_ptr;
    for(size_t i = 0; i < n; ++i)
    {
        __m128d c= _mm_mul_pd(a_ptr2[i], w_ptr2[i]);
         sum2 = _mm_add_pd(c, sum2);
    }
    sum_val = sum_ptr[0] + sum_ptr[1];
}

// Умножение и суммирование операндов с плавающим знаком одинарной точности (SSE-128 бит)
__attribute__((target("sse4.2"))) inline void dot_product_optimized_SSE128(float* a_ptr, float* w_ptr, size_t n, float& sum_val)
{
    __m128 sum4=_mm_setzero_ps();
    float*sum_ptr=(float*)(&sum4);
    __m128*a_ptr4=(__m128*)a_ptr;
    __m128*w_ptr4=(__m128*)w_ptr;
    for(size_t i = 0; i < n; i++)
    {
        __m128 c=_mm_mul_ps(a_ptr4[i],w_ptr4[i]);
         sum4 =_mm_add_ps(c, sum4);
    }
    sum_val = sum_ptr[0] + sum_ptr[1] + sum_ptr[2] + sum_ptr[3];
}

// Умножение и суммирование целочисленных 32-разрядных операндов(SSE-128 бит)
__attribute__((target("sse4.2"))) inline void dot_product_optimized_SSE128(int32_t* a_ptr, int32_t* w_ptr, size_t n, int32_t& sum_val)
{
    __m128i sum4 = _mm_setzero_si128();
    int32_t* sum_ptr = (int32_t *) &sum4;
    __m128i* a_ptr4 = (__m128i*)a_ptr;
    __m128i* w_ptr4 = (__m128i*)w_ptr;
    for(size_t i = 0; i < n; ++i)
    {
        __m128i c= _mm_mullo_epi32(a_ptr4[i], w_ptr4[i]);
        sum4 = _mm_add_epi32(c, sum4);
    }
    sum_val = sum_ptr[0] + sum_ptr[1] + sum_ptr[2] + sum_ptr[3];
}

// Выполнение многоканальной свёртки с расчётом времени
template<typename T>
float SSEConv::Convolution()
{
    // Одноканальный тензор с добавлением padding если требуется
    std::vector<T> pad_tensor;
    FillPaddingTensor(pad_tensor);

    // Размеры выходной матрицы с учётом padding
    int ResHeight =1 + (tensor_height_ - (padding_horiz_ ? 1 : kernel_height_)) / stride_vert_;
    int ResWidth = 1 + (tensor_width_ - (padding_horiz_ ? 1 : kernel_width_)) / stride_horiz_;

    // Выходные матрицы
    std::vector<std::vector<T> > pad_results(outputs_);

    // Временный вектор для хранения данных ядра
    std::vector<std::vector<char> > tempkernel(outputs_);

    temptensor_.resize(GetKernelSize() * inputs_ + kMAX_OPT_BYTES, 0);

    for (int o = 0; o < outputs_; ++o)
    {
        // Подготовка данных ядра - скопировать со всех входов
        tempkernel[o].resize(GetKernelSize() * inputs_ + kMAX_OPT_BYTES, 0);
        std::memcpy(tempkernel[o].data(), &(kernel_.data()[o * inputs_ * GetKernelSize()]), inputs_ * GetKernelSize());
    }

    // Выделить память для новых тензоров
    for (int i = 0; i < pad_results.size(); ++i)
        pad_results[i].resize(ResHeight * ResWidth);

    int steps = 0;
    float timer = 0;

    FixTime();
    do
    {
        for (int o = 0; o < outputs_; ++o)
            // Заполнить результирующий тензор нулями
            std::memset(pad_results[o].data(), 0, pad_results[o].size() * sizeof(T));

        // Один проход расчёта свёртки для каждого выхода
        for (int o = 0; o < outputs_; ++o)
            MakeConv<T>(pad_results[o].data(), pad_tensor.data(), (T *)tempkernel[o].data());

        if (accuracycheck_)
            break;

        // Продолжаем пока время расчётов не превысит 1 секунду
        timer = GetTime();
        ++steps;
    } while (timer < 1.0);

    // Проверка результатов с эталоном - если требуются
    if (accuracycheck_)
    {
        T *newres = (T *) pad_results[0].data();
        T *fixres = (T *) result_.data();
        float maxerror = 0;

        for (int i = 0; i < GetResultSize(false); ++i)
        {
            float curerror = 0;

            if (newres[i] && fixres[i])
                curerror = std::abs(((float)fixres[i] - (float)newres[i])/std::max((float)std::abs(fixres[i]), (float)std::abs(newres[i])));

            if (maxerror < curerror)
                maxerror = curerror;
        }
        return maxerror;
    }

    // При верификации сравнение с неоптимизированным расчётом
    if (verify_outs_)
    {
        ConvData testConv = *this;
        testConv.outputs_ = 1;

        // Тестовая свёртка для проверки
        auto test = BaseConv::CreateConv(OPT_TYPE::NONE, &testConv);
        test->FillData(tensor_.data(), GetTensorSize(), kernel_.data(), testConv.GetKernelSize() * testConv.inputs_);
        test->SetResult((const char *)pad_results[0].data(), testConv.GetResultSize(), true);
        verify_result_ = test->RunConv();
    }

    return ((float)steps / timer);
}

// Корректировка суммы нужна только для char
template<typename T>
inline void CorrectSum(T &val)
{
   ;
}

inline void CorrectSum(char &val)
{
    *((int *)&val) >>= 8;
}

// Свёртка одного канала
template<typename T>
int SSEConv::MakeConv(T *result, T *tensor, T *kernel, dummyidentity<T>)
{
    // Размеры выходной матрицы с учётом дополнения
    int ResHeight =1 + (tensor_height_ - (padding_horiz_ ? 1 : kernel_height_)) / stride_vert_;
    int ResWidth = 1 + (tensor_width_ - (padding_horiz_ ? 1 : kernel_width_)) / stride_horiz_;

    // Размеры тензора с учётом дополнения
    int pad_tensor_width = tensor_width_ + (padding_horiz_ ? (kernel_width_ - 1) : 0);

    // Временный массив для хранения данных тензора после сдвига
    T *tt = (T *)temptensor_.data();

    // Указатель на конец данных тензора временного массива после сдвига
    T *endtt = tt + GetKernelSize(false) * inputs_;

    // Размер ядра свёртки в байтах / 128 бит с округлением вверх
    size_t ks = GetKernelSize(true) * inputs_;
    ks = (ks % 16) ? (ks >> 4) + 1 : (ks >> 4);

    // Размер одного тензора в элементах
    int tensor_elems = tensor_width_ * tensor_height_;

    int stinc = stride_vert_ * pad_tensor_width;

    T* cursum = result;
    int64_t tempSum = 0;
    T &sum = (T &)tempSum;

    for (int y = 0; y < ResHeight; ++y)
    {
        // Верхний ряд в тензоре, участвующий в свёртке
        T *tensor_rowup = &tensor[y * stinc];

        for (int x = 0; x < ResWidth; ++x)
        {
            // Временный массив для хранения данных тензора после сдвига
            T *a = tt;

            // Верхний ряд в тензоре, смещённый относительно положения вычисляемого элемента
            T *tensor_upcur = &tensor_rowup[x * stride_horiz_];

            for (int i = 0; i < inputs_; ++i)
            {
                // Текущий ряд ядра свёртки, смещённый относительно положения вычисляемого элемента
                T *tensor_cur = &tensor_upcur[i * tensor_elems];

                for (int h = 0; h < kernel_height_; ++h)
                {
                    // Копирование подстроки тензора во временный массив
                    std::memcpy(a, tensor_cur, kernel_width_ * sizeof (T));

                    // Следующая строка матрицы
                    a += kernel_width_;
                    tensor_cur += pad_tensor_width;
                }
            }

            // Суммирование произведений матрицы ядра и подматрицы тензора
            dot_product_optimized_SSE128(tt, (T *)kernel, ks, sum);

            // Корректировка суммы для char
            CorrectSum(sum);
            *cursum += sum;
            tempSum = 0;
            ++cursum;
        }
    }

    return 0;
}
