struct Point {
  int x;
  int y;
};

void Point.move(int dx, int dy) {
  this->x += dx;
  this->y += dy;
}

int Point.sum(void) {
  return this->x + this->y;
}

union Word {
  int i;
  half h;
};

int main() {
  struct Point p;
  p.x = 3;
  p.y = 4;
  p.move(10, 20);
  print_int(p.sum());
  putc(10);
  union Word w;
  w.i = 65;
  putc(w.i);
  putc(10);
  w.h = 2.5h;
  int t = w.h;
  print_int(t);
  putc(10);
  return 0;
}
