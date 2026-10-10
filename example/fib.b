int n = 10;
int main() {
  int a = 0;
  int b = 1;
  int i = 0;
  while (i < n) {
    int t = a + b;
    a = b;
    b = t;
    i += 1;
  }
  print_int(a);
  putc(10);
  int s = 0;
  for (i = 1; i <= 10; i += 1) {
    if (i % 2 == 0) {
      continue;
    }
    if (s > 100) {
      break;
    }
    s += i;
  }
  print_int(s);
  putc(10);
  return 0;
}
