The example is designed to calculate the dot product of two vectors stored in binary files. \
The example provides various options for controlling this process, including the choice of data types (char, int32, float, or double) and the use of processor-level optimizations.

Usage: vecmulsm [KEY] [KEY]\
Calculates the sum of vector products from binary files with the following parameters:\
-h, --help, -? - Display this help message and exit.\
-q, -quiet - Quiet mode. Only the calculation result is displayed.\
-t=Z, -type=Z - Select the data type to be processed. Allowable data types:\
• f64 - double (64-bit floating-point),\
• f32 - float (32-bit floating-point),\
• i32 - int (32-bit integer),\
• i8 - char (8-bit integer).\
-i=filename, -infile=filename - Select input files (filename).\
Two file names must be provided (in two parameters), each containing vector data of the desired type.\
Example:
```bash
vecmulsm -t=i8 -i=file1.bin -i=file2.bin
```


Пример предназначен для вычисления скалярного произведения двух векторов, которые хранятся в бинарных файлах. \
Пример предоставляет различные опции для управления этим процессом, включая выбор типа данных (char, int32, float или double) и использование оптимизаций на уровне процессора.

Использование: vecmulsm [КЛЮЧ] [КЛЮЧ]\
Выдаёт сумму произведений векторов, заданных в бинарных файлах, заданных следующими параметрами:\
    -h, --help, -? - вызов данной справки с последующим выходом.\
    -q, -quiet — тихий режим. Выводится только результат расчёта.\
    -t=Z, -type=Z - выбор обсчитываемого типа данных. Допустимые типы данных:\
        • f64 — double (знаковое с плавающей точкой 64 bit),\
        • f32 — float (знаковое с плавающей точкой 32 bit),\
        • i32 — int (знаковое целое 32 bit),\
        • i8 — char (знаковое целое 8 bit).\
    -i=filename, -infile=filename — выбор входных файлов (filename). \
        Должны быть указаны два имени (в двух параметрах) бинарных файлов,\
        в каждом из которых сохранены данные вектора нужного типа.\
    Пример:
  ```bash
  vecmulsm -t=i8 -i=file1.bin -i=file2.bin
  ```
