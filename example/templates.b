template <typename T>
T max(T a, T b) {
  if (a > b) {
    return a;
  }
  return b;
}

template <typename T>
struct Box {
  T val;
};

template <typename T>
T unbox(struct Box<T> *b) {
  return b->val;
}

int main() {
  print_int(max<int>(3, 4));
  putc(10);
  half h = max<half>(1.5h, 2.5h);
  int ih = h;
  print_int(ih);
  putc(10);
  struct Box<int> b;
  b.val = 7;
  print_int(unbox<int>(&b));
  putc(10);
  return 0;
}
