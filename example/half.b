half pi = 3.5h;

half sq(half x) {
  return x * x;
}

int main() {
  half a = 1.5h;
  half b = a + 2;
  int ib = b;
  print_int(ib);
  putc(10);
  print_int(sq(3.0h) == 9.0h ? 1 : 0);
  putc(10);
  print_int(a < b ? 1 : 0);
  putc(10);
  half nan = 0.0h / 0.0h;
  print_int(nan != nan ? 1 : 0);
  putc(10);
  int ip = pi;
  print_int(ip);
  putc(10);
  return 0;
}
