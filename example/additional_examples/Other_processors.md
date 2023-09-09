**Пример реализации под другие процессоры**

Для реализации оптимизаций (к примеру под отличные от x86 процессоры) существующий код может быть дополнен ещё одним классом, назовём его «NewProcConv». \
Для его использования в перечисление  OPT_TYPE следует добавить новый тип оптимизации NEWPROC, а в функцию BaseConv::CreateConv(OPT_TYPE opt_type, CONV_TYPE type ... дополнительное условие:
```cpp
    case OPT_TYPE::NEWPROC:
       newConv = std::make_shared<NewProcConv>(type, tensor_height, tensor_width, height, width, inputs, outputs, stride_vert, stride_horiz, padding_vert, padding_horiz);
        break;
```
и в функцию BaseConv::CreateConv(OPT_TYPE opt_type, ConvData *conv)
```cpp
    case OPT_TYPE::NEWPROC:
	  newConv = std::make_shared<NewProcConv>(conv);
       break;
```

Описание создания класса «NewProcConv»:
1. Наследование от BaseConv: Класс будет наследоваться от BaseConv, таким образом, у него будет доступ к защищенным членам BaseConv.
2. Конструкторы: Нужно два конструктора: первый, который принимает указатель ConvData и второй, который принимает все необходимые параметры для создания объекта.
3. Метод RunConv: Это чисто виртуальная функция в BaseConv, поэтому она должна быть переопределена в классе. Этот метод будет использоваться для выполнения свёртки. Описание алгоритма реализации свёртки представлено в соответствующих документах.
4. Вспомогательные методы и переменные: Методы, такие как Convolution и MakeConv, будут приватными и использоваться для внутренней обработки. Кроме того,  возможно понадобится переменная для временного хранения данных, а также переменная для сохранения результата верификации.

Примерный шаблон для нового класса («NewProcConv.h»):
```cpp
class NewProcConv : public BaseConv
{
public:
    // Конструкторы
    NewProcConv(ConvData *conv);
    NewProcConv(CONV_TYPE type, int tensor_height, int tensor_width, int height, int width, 
                     int inputs = 1, int outputs = 1, int stride_vert = 1, int stride_horiz = 1, 
                     bool padding_vert = false, bool padding_horiz = false);

    // Переопределенный метод для выполнения свёртки
    float RunConv() override;

private:
    // Вспомогательные методы для выполнения свёртки
    template<typename T>
    float Convolution();

    template<typename T>
    int MakeConv(T *res, T *tensor, T *kernel) {
        return MakeConv(res, tensor, kernel, dummyidentity<T>());
    };

    template<typename T>
    int MakeConv(T *res, T *tensor, T *kernel, dummyidentity<T>);

    // Приватные переменные для внутреннего использования
    std::shared_ptr<aligned_char> temptensor_;
    float verify_result_;
};
```

Чтобы реализовать функциональность этого класса, потребуется определить каждую функцию (метод) в соответствующем файле реализации .cpp и интегрировать новый класс в приложение (подправив CmakeLists.txt — добавив в add_executable имена файлов нового класса и добавив #include "NewProcConv .h" в baseconv.cpp). 

Также для активации нового кода на соответствующих процессорах в функцию CheckOptimizationCPU необходимо добавить проверку на поддерживаемые векторные операции, используемые в новом классе и возвращать из функции OPT_TYPE::NEWPROC при условии их наличия.
