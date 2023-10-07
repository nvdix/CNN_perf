#!/bin/bash

# Проверка наличия аргументов командной строки
if [ "$#" -eq 0 ]; then
    echo "Ошибка: необходимо указать хотя бы один аргумент."
    exit 1
fi

# Инициализация переменной для хранения общего числа операций в секунду
total_time_inv=0.0

# Функция для вызова утилиты и добавления результатов к общему числу
calculate_ops() {
    local raw_layer_results=$(./convbench -q "$@")
    echo "Слои: $@"
    IFS=$'\n'
    for layer_result in $raw_layer_results; do
        layer_ops=$(echo "$layer_result" | awk -F': ' '{print $2}')
        echo "    $layer_ops операций в секунду"
        layer_time_inv=$(echo "1 / $layer_ops" | bc -l)
        total_time_inv=$(echo "$total_time_inv + $layer_time_inv" | bc -l)
    done
}

# Вызов утилиты с параметрами из аргументов командной строки
calculate_ops "$@"

# Вывод общего числа операций в секунду
total_ops=$(echo "1 / $total_time_inv" | bc -l)
echo "Операций в секунду по всем слоям: $total_ops"
