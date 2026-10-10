# bob16

A 16-bit CPU emulator (`bob16.c`), two-pass assembler (`bob16asm.py`), and
original-syntax converter (`bob16conv.py`).
Fork of https://github.com/somerandomviolinkid/bob16 with an extended mode-2
instruction set. Version 1.2, 59 mode-2 opcodes.

## Machine

* 16-bit words, 64K-word memory. Programs are raw little-endian binaries
  loaded at address `0x0000` with `pc = 0` (`./bob16 program.bin`).
* Dual instruction set, selected by `rs0` bit 0. The CPU boots in mode 1.
  `trap ext` / `enter2 label` (with `r7` = target) enters mode 2;
  `uext` leaves it.
* Memory map: user code+data in `0x0000..0x7FFF`; `0x8000..0xFFFF` is
  reserved for libc's own use (stack grows down from `0xFFFF`, plus future
  heap and libc statics). Enforced at build time, not hardware.
* Condition flags N/Z/P plus carry/borrow C (`adc`/`sbb`/`neg`/`rcr`,
  `jc`/`jnc`, `r1 = r1 + r2 + C` style extended arithmetic).
* `id rd, sel` introspection (cpuid-like): version, `FEAT_*` bit mask,
  memory top, register counts, opcode count, existing ports, `"BOB-16"` name.

## Mode 1 (16-bit instructions, r0–r7)

`nop add and not ld ldi ldr st sti str br[n][z][p] jmp ret jsr jsrr lea trap`
with traps `halt putc puts gets ext`, pseudos `mov clr li enter2`, and macros
`inc dec neg sub tst shl push pop`. `ld/st/lea/br/jsr` are pc-relative.

## Mode 2 (32/48-bit instructions, r0–r15, entered via `trap ext`)

Append `.cc` to update the condition codes, e.g. `add.cc`.

* ALU: `add sub mul and or xor not neg shl shr sar div mod sdiv smod
  adc sbb mulh mulhs rol ror rcr bset bclr`
  (`mulh`/`mulhs` = upper 16 of the product, unsigned/signed;
  `divmod rq, rn, rd` writes quotient to `rq` and remainder back into `rn`).
* Flags-only: `cmp tst btst` (`btst` also copies the tested bit into C).
* Jumps: `jmp jn jz jp jnz jle jge jc jnc`, `jal rd, target` (link in a
  register), `call rsp, target` / `ret rsp+off` (stack, with cleanup offset),
  `cbz`/`cbnz rs, target` (flag-preserving), `tbz`/`tbnz rs, bit, target.
* Memory: `load load2 stor stor2 ldb stb`; bytes are addressed as
  `word = addr>>1`, even address = low byte (`ldb` zero-extends,
  `stb` preserves the other half).
* Moves/system: `move swap in out id push pop peek pushm popm`
  (`pop rd, rsp+off` peeks ahead, `peek` never pops,
  `pushm`/`popm rsp, mask` save/restore every register in a bit mask).
* x86/ARM-inspired: `ldx`/`stx` (base+index), `copy`/`fill` (block ops),
  `rev`/`clz`/`ctz`/`popcnt`, `min`/`max` (signed), `cselz`/`cseln`/`cselc`
  (branchless select), `andn`, `btc`, `enter`/`leave` (stack frames),
  `getcc`/`setcc` (flag save/restore, bits N=8,Z=4,P=2,C=1).
* Operands: 4-bit short immediates (`0..15`, flag `i2`), 16-bit immediates
  (flag `i3`, automatic), `[reg]` memory operands, and `rN+/-off` /
  `[rN+/-off]` 8-bit offsets (flag `i4/i5`, `-255..255`, src = low byte,
  dst = high byte, shared sign). No offset on the last ALU operand.
  The low nibble of the second instruction word must be zero; its lowest
  bit is reserved as the next flag pool.

## I/O

* Port `0x00`: console character. Ports `0x70..0x73`: 64-bit virtual/real
  time (read with `in`, set with `out`). Mode 1 traps `putc`/`puts`/`gets`.
* Division by zero and invalid encodings trap with an address dump.

## Assembler

```
bob16asm.py prog.asm [-o prog.bin] [-l prog.lst] [--big-endian] [--limit 0x8000]
```

Two passes with labels, `+ - * / % << >> & | ^ ~ ( )` expressions, `$`
(address of the current line), registers `r0..r15` / `lr = r7`, binary16
literals (`1.5h -2.0h infh nanh`), and
directives `.org .mode .word .string/.asciz/.ascii .fill/.space/.blkw
.equ` (also `NAME = expr`). `ID_*` selectors and `FEAT_*` mask bits are
predefined. Mode-1 and mode-2 instructions are cross-checked
(`.mode 1|2` switches).

`--limit` caps the top of user memory (exclusive); use `0x8000` to keep
programs out of the libc-reserved upper half. `bobc` programs build with it:
`bobc.py p.b -o p.asm && bob16asm.py p.asm --limit 0x8000`.


## Compiler (`bobc`)

`bobc.py` compiles a C-like subset (`int`/`unsigned`/`char`/`half`/`void`, globals,
functions with params/calls/recursion, arrays + pointers, `if`/`while`/
`do`/`for`/`return`/`break`/`continue`, full C expression precedence,
`puts`/`putc`/`print_int`/`gettime`/`settime` (64-bit virtual time <->
four words at an address); see `example/hello.b`, `example/fib.b`,
`example/funcs.b`, `example/half.b`, `example/time.b`,
`example/uselib.b` + `example/mathlib.b`).
Args go right-to-left on the stack
(`r10` = SP, `r11` = FP, result in `r0`, caller cleans up). `half` is IEEE
binary16 (`1.5h` literals, `fdiv` never traps, `ftoi` traps out of range /
on NaN, NaN comparisons are all false except `!=`). Structs (`struct P {
... };`, `p.x` / `p->x`, struct assignment copies; params/returns must be
pointers) and enums (`enum C { RED, GREEN = 5 };` → int constants) round
out the type system; see `example/structs.b`. Methods (`void P.move(int dx)
{ this->x += dx; }`, called as `p.move(1, 2)` / `q->move(1, 2)` with `this`
as a `P *`) and unions (`union W { int i; half h; };`, all members at
offset 0, size = max) complete aggregates; see `example/methods.b`.
Templates (`template <typename T> T max(T a, T b) {...}`, used as
`max<int>(3, 4)`; also `struct`/`union` templates like
`struct Box<T> { T val; };`) monomorphize at parse time, so each instance
is an ordinary function/struct afterwards; see `example/templates.b`.
Explicit type arguments are required (no deduction yet), as is
definition-before-use.
Defer (`defer f(x);` or `defer { ... }`) runs code when the enclosing
scope ends, LIFO — including `return`/`break`/`continue` and each loop
iteration; loop bodies and branch blocks are scopes. Deferred code is
late-bound (it reads variables as they are when it runs) and shares the
function frame; `return` is banned inside it. See `example/defer.b`.

## Converter

```
bob16conv.py prog.basm [-o prog.asm] [-a] [-q]
```

Converts original bob16 assembly (decimal offsets, `.fill HEX`,
`.stringz`) into this assembler's syntax, preserving addresses, and warns
where the original assembler disagrees with the documented semantics
(`nop`, `jsr`/`jsrr`, in-place `add`/`and`, `add` with immediate).
