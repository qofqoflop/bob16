void work(int x) {
  defer putc(69);
  if (x > 0) {
    defer putc(70);
    print_int(x * 2);
    putc(10);
    return;
  }
  print_int(x);
  putc(10);
}

int main() {
  work(21);
  work(-5);
  int i = 0;
  while (i < 3) {
    defer putc(48 + i);
    i += 1;
  }
  putc(10);
  {
    defer putc(65);
    defer {
      putc(66);
      putc(67);
    }
    putc(68);
  }
  putc(10);
  return 0;
}
