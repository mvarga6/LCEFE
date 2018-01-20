#include "../include/logger.h"

using namespace std;

// string formatted(const char* format, ...)
// {
//     string result;
//     va_list args;
//     va_start(args, format);
//     result = formatted(format, args);
//     va_end(args);
//     return result;
// }

// string formatted(const char* format, va_list args)
// {
// #ifndef _MSC_VER
//     size_t size = snprintf( nullptr, 0, format, args) + 1; // Extra space for '\0'
//     unique_ptr<char[]> buf( new char[ size ] ); 
//     vsnprintf( buf.get(), size, format, args);
//     return string(buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
// #else
//     int size = _vscprintf(format, args);
//     string result(++size, 0);
//     vsnprintf_s((char*)result.data(), size, _TRUNCATE, format, args);
//     return result;
// #endif
// }

#include <cstdarg>    // va_start, va_end, std::va_list
#include <cstddef>    // std::size_t
#include <stdexcept>  // std::runtime_error
#include <vector>     // std::vector



string formatted(const char *const format, ...)
{
  auto temp = std::vector<char> {};
  auto length = std::size_t {63};
  std::va_list args;
  while (temp.size() <= length)
    {
      temp.resize(length + 1);
      va_start(args, format);
      const auto status = std::vsnprintf(temp.data(), temp.size(), format, args);
      va_end(args);
      if (status < 0)
        throw std::runtime_error {"string formatting error"};
      length = static_cast<std::size_t>(status);
    }
  return std::string {temp.data(), length};
}