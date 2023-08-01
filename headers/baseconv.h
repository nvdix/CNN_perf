#ifndef BASECONV_H
#define BASECONV_H

#include "consts.h"
#include "errors.h"
#include "utils.h"

// Базовый класс описывающий данные свёртки
class ConvData
{
public:
    ConvData();
    int tensor_height_;
    int tensor_width_;
    int kernel_height_;
    int kernel_width_;
    int inputs_;
    int outputs_;
    int stride_horiz_;
    int stride_vert_;
    bool padding_horiz_;
    bool padding_vert_;
    // Текущий тип
    CONV_TYPE conv_type_;
    // Описание свёртки
    std::string note_;

    int GetTensorSize(bool inbytes = true);
    int GetKernelSize(bool inbytes = true);
    int GetTypeSize();

    int GetResultSize(bool inbytes = true);

    // Получить строку с используемым типом данных
    const char * GetTypeinString();

    // Верификация выхода
    bool verify_outs_;
};

// Структура для обработки шаблонов
template<typename T>
struct dummyidentity { typedef T type; };

// Базовый класс расчёта свёртки
class BaseConv : protected ConvData
{
public:
    BaseConv();
    // Создание нового объекта свёртки
    static std::shared_ptr<BaseConv> CreateConv(OPT_TYPE opt_type, CONV_TYPE conv_type, int tensor_height, int tensor_width, int kernel_height, int kernel_width,
                                                int inputs = 1, int outputs = 1, int stride_vert = 1, int stride_horiz = 1, bool padding_vert = false, bool padding_horiz = false);

    static std::shared_ptr<BaseConv> CreateConv(OPT_TYPE opt_type, ConvData *conv);

    // Задать данные
    int FillData(const char *tensor = nullptr, int tensor_size = 0, const char *kernel = nullptr, int kernel_size = 0);

    // Расчёт свёртки
    // Возвращает число рассчитанных за секунду свёрток
    // cmpdata [in] - выходные данные для проверки результатов
    virtual float RunConv() = 0;

    // Зафиксировать начальное время таймера расчёта задержки
    void FixTime();

    // Получить время от старта таймера в секундах
    float GetTime();

    void SetNote(const std::string str) {note_ = str;};

    // Задать сравнение с результатом
    void SetResult(const char *data, int size, bool quiet=false);
protected:
    // Время начала расчёта свёртки
    std::chrono::high_resolution_clock::time_point timestart_;

    // Входные данные тензора
    std::vector<char> tensor_;

    // Данные свёрточных ядер
    std::vector<char> kernel_;

    // Итоговые данные для проверки точности
    std::vector<char> result_;

    // Проверка точности
    bool accuracycheck_;

    // Проверка результата с выводом сообщений или без
    bool quietcheck_;

    // Перестроить тензор в одноканальный (с добавлением padding если требуется)
    template<typename T>
    void FillPaddingTensor(std::vector<T> &tensvector)
    {
        // Размеры одноканального тензора с учётом padding
        int pad_tensor_height = tensor_height_ + (padding_vert_ ? (kernel_height_ - 1) : 0);
        int pad_tensor_width = tensor_width_ + (padding_horiz_ ? (kernel_width_ - 1) : 0);
        int pad_tensor_size = pad_tensor_height * pad_tensor_width;

        // Выделить память для новых тензоров = 0
        tensvector.clear();
        tensvector.resize(pad_tensor_height * pad_tensor_width * inputs_, (T)0);

        auto cc = (T*)tensor_.data();

        for (int t = 0; t < inputs_; ++t)
        {
            // Текущая строка в тензоре
            int tensor_ncurrow = 0;
            for (int y = 0; y < pad_tensor_height; ++y)
            {
                if (padding_vert_ && ((y < (kernel_height_ >> 1)) || (y >= tensor_height_ + (kernel_height_ >> 1))))
                    continue;

                T *tensor_currow = ((T*)tensor_.data()) + tensor_ncurrow++ * tensor_width_ * inputs_;
                T *one_tensor_currow = padding_horiz_ ? &tensvector[pad_tensor_size * t + y * pad_tensor_width + (kernel_width_ >> 1)] :
                        &tensvector[pad_tensor_size * t + y * pad_tensor_width];

                // Заполнение одноканального тензора
                for (int x = 0; x < tensor_width_; ++x)
                    one_tensor_currow[x] = tensor_currow[x * inputs_ + t];
            }
        }
    };

};

#endif // BASECONV_H
