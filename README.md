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
bob16asm.py prog.asm [-o prog.bin] [-l prog.lst] [--big-endian]
```

Two passes with labels, `+ - * / % << >> & | ^ ~ ( )` expressions, `$`
(address of the current line), registers `r0..r15` / `lr = r7`, binary16
literals (`1.5h -2.0h infh nanh`), and
directives `.org .mode .word .string/.asciz/.ascii .fill/.space/.blkw
.equ` (also `NAME = expr`). `ID_*` selectors and `FEAT_*` mask bits are
predefined. Mode-1 and mode-2 instructions are cross-checked
(`.mode 1|2` switches).

## Converter

```
bob16conv.py prog.basm [-o prog.asm] [-a] [-q]
```

Converts original bob16 assembly (decimal offsets, `.fill HEX`,
`.stringz`) into this assembler's syntax, preserving addresses, and warns
where the original assembler disagrees with the documented semantics
(`nop`, `jsr`/`jsrr`, in-place `add`/`and`, `add` with immediate).
