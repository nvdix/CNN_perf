#!/usr/bin/env python3
import numpy as np

def save_random_int8_to_file(file_name, size):
    # Create an array of random numbers of type int8
    random_int8 = np.random.randint(low=0, high=2, size=size, dtype=np.int8)

    # Saving the array to a binary file
    random_int8.tofile(file_name)


def save_random_int32_to_file(file_name, size):
    # Create an array of random numbers of type int32
    random_int32 = np.random.randint(low=0, high=3, size=size, dtype=np.int32)

    # Saving the array to a binary file
    random_int32.tofile(file_name)


def save_random_float_to_file(file_name, size):
    # Create an array of random numbers of type float
    random_float = np.random.uniform(low=0, high=10, size=size).astype(np.float32)

    # Saving the array to a binary file
    random_float.tofile(file_name)


def save_random_double_to_file(file_name, size):
    # Create an array of random numbers of type double
    random_double = np.random.uniform(low=0, high=10, size=size).astype(np.float64)

    # Saving the array to a binary file
    random_double.tofile(file_name)


# Usage
save_random_int8_to_file("random_int8_1.bin", 64)
save_random_int8_to_file("random_int8_2.bin", 64)

save_random_int32_to_file("random_int32_1.bin", 16)
save_random_int32_to_file("random_int32_2.bin", 16)

save_random_float_to_file("random_float_1.bin", 16)
save_random_float_to_file("random_float_2.bin", 16)

save_random_double_to_file("random_double_1.bin", 8)
save_random_double_to_file("random_double_2.bin", 8)
