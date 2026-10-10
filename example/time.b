int t[4];

int main() {
  gettime(t);
  print_int(t[1]);
  putc(10);
  print_int(t[0]);
  putc(10);
  t[0] = 0;
  t[1] = 0;
  t[2] = 0;
  t[3] = 0;
  settime(t);
  gettime(t);
  print_int(t[3] + t[2] + t[1] + t[0]);
  putc(10);
  return 0;
}
