struct Point {
  int x;
  int y;
};

enum Color {
  RED,
  GREEN = 5,
  BLUE
};

int plen(struct Point *p) {
  return p->x + p->y;
}

int main() {
  struct Point p;
  p.x = 3;
  p.y = 4;
  print_int(plen(&p));
  putc(10);
  struct Point q = p;
  q.x += GREEN;
  print_int(q.x + q.y + RED + BLUE);
  putc(10);
  return 0;
}
