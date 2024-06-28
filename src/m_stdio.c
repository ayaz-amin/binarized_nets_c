#define STB_SPRINTF_IMPLEMENTATION
#include <windows.h>
#include "stb_sprintf.h"

#include "m_stdio.h"

int m_printf(const char *fmt, ...)
{
    HANDLE stdout = GetStdHandle(STD_OUTPUT_HANDLE);
    va_list arg;
    int count;
    va_start(arg, fmt);
    char buff[256];
    count = stbsp_vsprintf(buff, fmt, arg);
    va_end(arg);
    WriteConsole(stdout, buff, count, 0, 0);
    return count;
}
