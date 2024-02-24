#include "signalflow/node/ml/Rave/utils.h"

#include <algorithm>
#include <cctype>
#include <sstream>

bool to_bool(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  std::istringstream is(str);
  bool b;
  is >> std::boolalpha >> b;
  return b;
}

int to_int(std::string str) { return stoi(str); }

float to_float(std::string str) { return stof(str); }

unsigned power_ceil(unsigned x) {
    if (x <= 1)
        return 1;
    int power = 2;
    x--;
    while (x >>= 1)
        power <<= 1;
    return power;
}
