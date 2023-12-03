#ifndef UPD_LOCALE_H
#define UPD_LOCALE_H

#ifndef _WIN32
#define UpdateEncoding(x) x
#else
#include "windows.h"
#include <string>

#define UpdateEncoding(x) UTF8toAnsi(x)
std::string UTF8toAnsi(const char* UTF8string);

#endif

#endif