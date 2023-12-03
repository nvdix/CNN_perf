#include "headers/updencoding.h"
#include <vector>
#include <locale>

#ifdef _WIN32

std::wstring UTF8toUNICODE(const char* UTF8string)
{
	if (!UTF8string)
		return L"";

	int UTF8stringLength = strlen(UTF8string);
	int size_needed = MultiByteToWideChar(CP_UTF8, 0, UTF8string, UTF8stringLength, NULL, 0);
	std::wstring wstrTo(size_needed, 0);
	MultiByteToWideChar(CP_UTF8, 0, UTF8string, UTF8stringLength, &wstrTo[0], size_needed);
	return wstrTo;
}

std::string UNICODEtoANSI(const WCHAR* UNICODEstring)
{
	if (!UNICODEstring)
		return "";

	size_t	wLen = wcslen(UNICODEstring);
	std::vector<char> ANSIstring(wLen + 1, 0);
	WideCharToMultiByte(CP_ACP, 0, UNICODEstring, wLen, &ANSIstring[0], wLen, NULL, NULL);
	return &ANSIstring[0];
}

std::string UTF8toAnsi(const char* UTF8string)
{
	return UNICODEtoANSI(UTF8toUNICODE(UTF8string).c_str());
}

#endif