The example is designed to calculate the dot product of two vectors stored in binary files. \
The example provides various options for controlling this process, including the choice of data types (char, int32, float, or double), and uses available processor-level optimizations. The automatic selection of the best optimization for the processor is performed in the multiplyAndSum() function of the CVectorMulSum class, defined in the "mulsum.h" file (lines 64 - 76). That is, AVX512 or, if it's not available, then AVX, otherwise SSE, and if no optimization is supported at all, then without it (using regular processor instructions).

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

To generate two binary files of different types, it is suggested to use the script _make_files.py_\
To run it, while in the _CNN_perf/example_ directory, execute:\
Installing the latest version of Python:
```bash
sudo apt install python3
```
Installing the pip package manager:
```bash
sudo apt install python3-pip
```
Installing the Python virtual environment package:
```bash
sudo apt install python3-venv
```
Creating a Python virtual environment:
```bash
python3 -m venv my_env
```
Activating the virtual environment:
```bash
source my_env/bin/activate
```
Installing numpy library
```bash
pip install numpy
```
Creating a pair of files:
```bash
python3 make_files.py
```

Пример предназначен для вычисления скалярного произведения двух векторов, которые хранятся в бинарных файлах. \
Пример предоставляет различные опции для управления этим процессом, включая выбор типа данных (char, int32, float или double), и использует доступные оптимизации на уровне процессора. Под процессор производится автоматический выбор наилучшей оптимизации в функции multiplyAndSum() класса CVectorMulSum, заданной в файле "mulsum.h" (строки 64 - 76). Т.е. AVX512 либо, если её нет, то AVX, иначе SSE, и если воообще не поддерживается никакая оптимизация, то без неё (используются обычные процессорные инструкции).

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

Для генерации двух бинарных файлов различных типов предлагается использовать скрипт _make_files.py_\
Для его запуска необходимо, находясь в каталоге _CNN_perf/example_, выполнить :\
Установка последней версии Python
```bash
sudo apt install python3
```
Установка менеджера пакетов pip
```bash
sudo apt install python3-pip
```
Установка пакета создания виртуальной среды Python
```bash
sudo apt install python3-venv
```
Создание виртуальной среды Python
```bash
python3 -m venv my_env
```
Активация виртуальной среды
```bash
source my_env/bin/activate
```
Установка библиотеки numpy
```bash
pip install numpy
```
Создание пар файлов
```bash
python3 make_files.py
```
