class Array(T) {
  int size;
  this(int i) {
    size = i;
  }
}

class Kernel {
  string code;
  this(string s) {
    code = s;
  }
}

auto rtc(string code) {
  return new Kernel(code);
}

void call(T...)(T ts) {
}
