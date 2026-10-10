int add(int a, int b) {
  return a + b;
}

int fact(int n) {
  if (n <= 1) {
    return 1;
  }
  return n * fact(n - 1);
}

void greet(void) {
  puts("hi");
}

int main() {
  greet();
  print_int(add(20, 22));
  putc(10);
  print_int(fact(5));
  putc(10);
  return 0;
}
