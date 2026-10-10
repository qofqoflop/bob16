#!/usr/bin/env python3
"""
bob16asm.py - assembler for the BOB-16 CPU (https://github.com/qofqoflop/bob16)

Usage:  bob16asm.py prog.asm [-o prog.bin] [-l prog.lst] [--big-endian]

Output is a raw binary of 16-bit little-endian words, word 0 = address 0,
which is exactly what the emulator's load_program() expects.

SYNTAX
  label:  mnemonic  op, op, op     ; comment   (also // comments)
  Registers: r0..r7 (mode 1), r0..r15 (mode 2); lr = r7
  Numbers:   42  #42  -5  0x2A  0b101  'a'  '\\n'      ($ = address of this line)
  Expressions: + - * / % << >> & | ^ ~ ( ) with labels and .equ symbols

DIRECTIVES
  .org  addr            set the location counter
  .mode 1|2             choose which instruction set the following code uses
  .word v, v, ...       16-bit words
  .string "text", ...   one char per word + 0 terminator (what `puts` wants)
  .ascii  "text"        same, no terminator
  .fill n [, v]         n words of v (default 0)      (.space / .blkw = same)
  .equ NAME, expr       or   NAME = expr

MODE 1 (16-bit instructions; the CPU boots in this mode)
  nop
  add/and rd, rs1, rs2        add/and rd, rs1, imm4 (-8..7)
  add/and rd, rs              add/and rd, imm7 (-64..63)
  not rd | not rd, rs
  ld / ldi / st / sti / lea   reg, label        (pc-relative, +-256 words)
  ldr / str  rd, rb, off6     or  rd, [rb+off]  (off -32..31)
  br[n][z][p] label           (plain `br` = always)
  jmp rd | ret | jsr label (+-1024 words) | jsrr rs
  trap n | halt | putc | puts | gets | ext
  Pseudo: mov rd, rs | clr rd | li rd, imm(-64..63) | enter2 label
  Macros (they set the condition codes like the instructions they expand to):
    inc rd | inc rd, rs        rd = rs + 1                              (1 word)
    dec rd | dec rd, rs        rd = rs - 1                              (1 word)
    neg rd | neg rd, rs        rd = -rs  (not; add 1)                   (2 words)
    tst rd                     set cc from rd                           (1 word)
    shl rd [, n]               rd <<= n (1..16) by doubling             (n words)
    sub rd, x | sub rd, rs1, x   x = register or immediate; no scratch  (1-3 words)
    push rsp, rv | pop rd, rsp   stack in memory, same layout as mode 2 (2 words)

MODE 2 (32-bit instructions, entered with `trap ext` with r7 = code address;
        left again with `uext`)
  Add `.cc` to a mnemonic to update the condition codes, e.g. add.cc
  Macros: inc dec neg clr tst (all take .cc and [reg] memory operands, e.g. inc [r1]),
          li = move, and any 3-operand ALU op may drop its first source:
          `add r1, 5` means `add r1, r1, 5`.
  A 16-bit immediate (flag i3) makes the instruction 3 words long.  It is used
  automatically wherever an immediate is accepted and is not a 0..15 literal.
  nop, halt
  uext rd|addr
  add sub mul and or xor shl shr sar div mod sdiv smod adc sbb   rd, rs1, rs2|imm
        (imm 0..15 uses the short form; anything else, negatives and forward
         label references included, uses a 16-bit immediate: add r1, r1, -1)
  not rd, rs      swap rd, rs     pop rd, rsp     ret rsp
  move rd, rs|imm       load rd, raddr|imm     load2 rd, raddr|imm
  stor raddr, rval|imm  stor2 raddr, rval|imm
  in rd, rport|imm      out rport, rval|imm    id rd, rsel|imm
  push rsp, rval|imm    call rsp, rtarget|imm
  (id selectors ID_MAX ID_VERSION ID_FEATURES ID_MEMTOP ID_RS0 ID_REGS ID_OPCODES
   ID_PORTS ID_NAME0..2 and bit masks FEAT_MEMOPS/IMM16/SDIV/IO/STACK/OFF are predefined)
  cmp rs1, rs2|imm
  jmp jn jz jp jnz jle jge jc  rtarget|addr          (jmp loop)
  Memory operands: add sub mul and or xor shl shr sar div mod sdiv smod cmp adc sbb
  accept [reg] for any operand: [rs1]/[rs2] read mem[reg] instead of reg, and a
  bracketed destination [rd] stores the result to mem[rd], e.g.
      add [r2], [r1], 5      ; mem[r2] = mem[r1] + 5
      add r4, r3, [0x2000]   ; absolute read: r4 = r3 + mem[0x2000]
      cmp [r1], [r2]
  Offsets (flag i4/i5, FEAT_OFF): append +off/-off (-255..255, shared sign)
  to a register base; src uses the low byte, dst the high byte, e.g.
      add r3, r1+5, r2       ; r3 = (r1+5) + r2
      add [r4+0x10], r1, r2  ; mem[r4+0x10] = r1 + r2
      load r1, r2+5          ; r1 = mem[r2+5]
      stor r4+0x10, r1+5     ; mem[r4+0x10] = r1+5
      jmp r1+5               ; pc = r1+5       (ret rsp+off cleans the stack)
      out r3+1, r1+5         ; port r3+1 gets r1+5
  No offset on the last ALU operand (`add r1, r2, r3+1` is rejected).
"""
import argparse
import ast
import operator
import re
import sys


class AsmError(Exception):
    pass


# --------------------------------------------------------------------------
# expression evaluation
# --------------------------------------------------------------------------
BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.LShift: operator.lshift, ast.RShift: operator.rshift,
    ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor,
}
UN_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert}
ESC = {'n': 10, 't': 9, 'r': 13, '0': 0, '\\': 92, "'": 39, '"': 34, 'a': 7, 'e': 27}


def _char_sub(m):
    s = m.group(1)
    if s[0] == '\\':
        if s[1] not in ESC:
            raise AsmError("unknown escape '\\%s'" % s[1])
        return str(ESC[s[1]])
    return str(ord(s))


class Env:
    def __init__(self, syms, final):
        self.syms = syms
        self.final = final
        self.pc = 0
        self.mode = 1
        self.item = None      # line being assembled (remembers its i3 decision)
        self.undef = False    # set when an operand used a not-yet-defined symbol


def eval_expr(text, env, strict=False):
    s = text.strip()
    if not s:
        raise AsmError("missing operand")
    s = re.sub(r"'(\\.|[^\\'])'", _char_sub, s)
    s = s.replace('#', '').replace('$', '__here__')
    s = re.sub(r'(?<!/)/(?!/)', '//', s)
    try:
        tree = ast.parse(s, mode='eval')
    except SyntaxError:
        raise AsmError("bad expression '%s'" % text.strip())

    def ev(n):
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return n.value
        if isinstance(n, ast.Name):
            if n.id == '__here__':
                return env.pc
            if n.id in env.syms:
                return env.syms[n.id]
            if not env.final and not strict:
                env.undef = True
                return 0
            raise AsmError("undefined symbol '%s'" % n.id)
        if isinstance(n, ast.BinOp) and type(n.op) in BIN_OPS:
            try:
                return BIN_OPS[type(n.op)](ev(n.left), ev(n.right))
            except ZeroDivisionError:
                raise AsmError("division by zero in expression")
        if isinstance(n, ast.UnaryOp) and type(n.op) in UN_OPS:
            return UN_OPS[type(n.op)](ev(n.operand))
        raise AsmError("bad expression '%s'" % text.strip())

    return ev(tree.body)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def chk_signed(env, v, bits, what):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if env.final and not lo <= v <= hi:
        raise AsmError("%s %d does not fit in %d signed bits (%d..%d)" % (what, v, bits, lo, hi))
    return v & ((1 << bits) - 1)


def chk_unsigned(env, v, bits, what):
    hi = (1 << bits) - 1
    if env.final and not 0 <= v <= hi:
        raise AsmError("%s %d out of range (0..%d)" % (what, v, hi))
    return v & hi


def rel_off(env, target):
    off = (target - (env.pc + 1)) & 0xFFFF
    return off - 0x10000 if off >= 0x8000 else off


def rel(env, target, bits):
    return chk_signed(env, rel_off(env, target), bits, "pc-relative distance")


REG_RE = re.compile(r'^(?:r(\d+)|(lr))$', re.I)


def as_reg(tok, maxreg):
    m = REG_RE.match(tok.strip())
    if not m:
        return None
    n = 7 if m.group(2) else int(m.group(1))
    if n > maxreg:
        raise AsmError("register r%d not available here (max r%d)" % (n, maxreg))
    return n


def reg(tok, maxreg):
    r = as_reg(tok, maxreg)
    if r is None:
        raise AsmError("expected a register, got '%s'" % tok.strip())
    return r


def need(mn, ops, n):
    if len(ops) != n:
        raise AsmError("'%s' takes %d operand(s), got %d" % (mn, n, len(ops)))


# --------------------------------------------------------------------------
# mode 1
# --------------------------------------------------------------------------
PCREL = {'ld': 4, 'ldi': 5, 'st': 7, 'sti': 8, 'lea': 13}
TRAPS = {'halt': 0, 'putc': 1, 'puts': 2, 'gets': 3, 'ext': 4}
M1_NAMES = {'nop', 'add', 'and', 'not', 'ldr', 'str', 'jmp', 'jsr', 'jsrr', 'ret',
            'trap', 'mov', 'clr', 'li', 'enter2',
            'inc', 'dec', 'neg', 'sub', 'tst', 'shl', 'push', 'pop'} | set(PCREL) | set(TRAPS)
BR_RE = re.compile(r'^br([nzp]*)$')


def parse_mem(mn, ops, env):
    if len(ops) == 2 and ops[1].strip().startswith('['):
        t = ops[1].strip()
        if not t.endswith(']'):
            raise AsmError("missing ']'")
        m = re.match(r'\s*(\w+)\s*(.*)$', t[1:-1])
        if not m:
            raise AsmError("bad memory operand '%s'" % t)
        base, rest = m.group(1), m.group(2).strip()
        off = '0' if not rest else (rest[1:] if rest[0] == ',' else rest)
    elif len(ops) == 3:
        base, off = ops[1], ops[2]
    elif len(ops) == 2:
        base, off = ops[1], '0'
    else:
        raise AsmError("'%s' wants: rd, rbase, offset   or   rd, [rbase+offset]" % mn)
    return reg(base, 7), eval_expr(off, env)


def enc1(env, mn, ops):
    R = lambda t: reg(t, 7)

    if mn == 'nop':
        need(mn, ops, 0)
        return [0]

    if mn in ('add', 'and'):
        op = 1 if mn == 'add' else 2
        if len(ops) == 3:
            rd, rs1 = R(ops[0]), R(ops[1])
            rs2 = as_reg(ops[2], 7)
            if rs2 is not None:
                return [op << 12 | rd << 9 | rs1 << 4 | rs2 << 1]
            v = eval_expr(ops[2], env)
            if rd == rs1 and not -8 <= v <= 7 and -64 <= v <= 63:   # use imm7 form
                return [op << 12 | rd << 9 | 3 << 7 | (v & 0x7F)]
            return [op << 12 | rd << 9 | 1 << 7 | rs1 << 4 | chk_signed(env, v, 4, "immediate")]
        if len(ops) == 2:
            rd = R(ops[0])
            rs = as_reg(ops[1], 7)
            if rs is not None:
                return [op << 12 | rd << 9 | 2 << 7 | rs << 4]
            v = eval_expr(ops[1], env)
            return [op << 12 | rd << 9 | 3 << 7 | chk_signed(env, v, 7, "immediate")]
        raise AsmError("'%s' takes 2 or 3 operands" % mn)

    if mn == 'not':
        if len(ops) == 1:
            return [3 << 12 | R(ops[0]) << 9 | 0x100]
        need(mn, ops, 2)
        return [3 << 12 | R(ops[0]) << 9 | R(ops[1]) << 5]

    if mn in PCREL:
        need(mn, ops, 2)
        rd = R(ops[0])
        return [PCREL[mn] << 12 | rd << 9 | rel(env, eval_expr(ops[1], env), 9)]

    if mn in ('ldr', 'str'):
        rd = R(ops[0]) if ops else None
        if rd is None:
            raise AsmError("'%s' needs operands" % mn)
        base, off = parse_mem(mn, ops, env)
        op = 6 if mn == 'ldr' else 9
        return [op << 12 | rd << 9 | base << 6 | chk_signed(env, off, 6, "offset")]

    m = BR_RE.match(mn)
    if m:
        flags = m.group(1) or 'nzp'
        bits = ('n' in flags) << 11 | ('z' in flags) << 10 | ('p' in flags) << 9
        need(mn, ops, 1)
        return [10 << 12 | bits | rel(env, eval_expr(ops[0], env), 9)]

    if mn == 'jmp':
        need(mn, ops, 1)
        return [11 << 12 | R(ops[0]) << 9]

    if mn == 'ret':
        need(mn, ops, 0)
        return [14 << 12]

    if mn == 'jsr':          # bit 11 clear = imm11 (signed), as in the original bob16
        need(mn, ops, 1)
        return [12 << 12 | rel(env, eval_expr(ops[0], env), 11)]

    if mn == 'jsrr':         # bit 11 set, register in bits 10:8
        need(mn, ops, 1)
        return [12 << 12 | 0x800 | R(ops[0]) << 8]

    if mn == 'trap':
        need(mn, ops, 1)
        t = ops[0].strip().lower()
        n = TRAPS[t] if t in TRAPS else eval_expr(ops[0], env)
        if env.final and not 0 <= n <= 4:
            raise AsmError("unknown trap vector %d (valid: 0..4)" % n)
        return [15 << 12 | (n & 0xF) << 8]

    if mn in TRAPS:
        need(mn, ops, 0)
        return [15 << 12 | TRAPS[mn] << 8]

    # ---- pseudo-instructions ----
    if mn == 'mov':
        need(mn, ops, 2)
        rd = R(ops[0])
        rs = as_reg(ops[1], 7)
        if rs is None:
            raise AsmError("mov takes two registers; use `li` for small constants")
        return [1 << 12 | rd << 9 | 1 << 7 | rs << 4]

    if mn == 'clr':
        need(mn, ops, 1)
        return [2 << 12 | R(ops[0]) << 9 | 3 << 7]

    if mn == 'li':
        need(mn, ops, 2)
        rd = R(ops[0])
        v = eval_expr(ops[1], env)
        if env.final and not -64 <= v <= 63:
            raise AsmError("li only handles -64..63; put the value in a .word and `ld` it")
        return [2 << 12 | rd << 9 | 3 << 7, 1 << 12 | rd << 9 | 3 << 7 | (v & 0x7F)]

    if mn == 'enter2':       # lea r7, label ; trap ext
        need(mn, ops, 1)
        return [13 << 12 | 7 << 9 | rel(env, eval_expr(ops[0], env), 9), 15 << 12 | 4 << 8]

    # ---- macros built from the instructions above ----
    def seq(*calls):
        out = []
        for m, o in calls:
            out += enc1(env, m, o)
        return out

    if mn in ('inc', 'dec'):            # inc rd | inc rd, rs      (1 word)
        if len(ops) not in (1, 2):
            raise AsmError("'%s' takes 1 or 2 operands" % mn)
        src = ops[1] if len(ops) == 2 else ops[0]
        return enc1(env, 'add', [ops[0], src, '1' if mn == 'inc' else '-1'])

    if mn == 'neg':                     # neg rd | neg rd, rs      (2 words)
        if len(ops) not in (1, 2):
            raise AsmError("'neg' takes 1 or 2 operands")
        return seq(('not', ops), ('add', [ops[0], '1']))

    if mn == 'tst':                     # set the condition codes from rd   (1 word)
        need(mn, ops, 1)
        return enc1(env, 'add', [ops[0], ops[0], '0'])

    if mn == 'shl':                     # shl rd [, n]: rd <<= n by doubling (n words)
        if len(ops) not in (1, 2):
            raise AsmError("'shl' takes a register and an optional count")
        n = eval_expr(ops[1], env, strict=True) if len(ops) == 2 else 1
        if not 1 <= n <= 16:
            raise AsmError("shl count must be 1..16 (each shift is one instruction)")
        return enc1(env, 'add', [ops[0], ops[0], ops[0]]) * n

    if mn == 'sub':                     # sub rd, x | sub rd, rs1, x     (1-3 words)
        if len(ops) == 2:
            ops = [ops[0], ops[0], ops[1]]
        need(mn, ops, 3)
        rd, rs1 = R(ops[0]), R(ops[1])
        rs2 = as_reg(ops[2], 7)
        if rs2 is None:                 # subtract an immediate
            v = eval_expr(ops[2], env)
            lo, hi = (-63, 64) if rd == rs1 else (-7, 8)
            if env.final and not lo <= v <= hi:
                raise AsmError("sub immediate %d out of range (%d..%d)" % (v, lo, hi))
            return enc1(env, 'add', [ops[0], ops[1], str(-v)])
        if rs1 == rs2:
            return enc1(env, 'clr', [ops[0]])
        if rd != rs2:                   # rd = ~(~rs1 + rs2)
            return seq(('not', [ops[0], ops[1]]), ('add', [ops[0], ops[0], ops[2]]),
                       ('not', [ops[0], ops[0]]))
        return seq(('not', [ops[0], ops[0]]), ('add', [ops[0], ops[0], ops[1]]),
                   ('add', [ops[0], '1']))      # rd == rs2: rs1 + ~rs2 + 1

    if mn == 'push':                    # push rsp, rv   (store, then decrement - like mode 2)
        need(mn, ops, 2)
        return seq(('str', [ops[1], ops[0], '0']), ('add', [ops[0], '-1']))

    if mn == 'pop':                     # pop rd, rsp    (increment, then load - like mode 2)
        need(mn, ops, 2)
        if R(ops[0]) == R(ops[1]):
            raise AsmError("pop: destination and stack register must differ")
        return seq(('add', [ops[1], '1']), ('ldr', [ops[0], ops[1], '0']))

    raise AsmError("internal: unhandled mode-1 mnemonic '%s'" % mn)


# --------------------------------------------------------------------------
# mode 2
# --------------------------------------------------------------------------
M2 = {
    'nop': (0, 'none'), 'halt': (1, 'none'), 'uext': (2, 'j'),
    'add': (3, 'rrx'), 'sub': (4, 'rrx'), 'mul': (5, 'rrx'),
    'and': (6, 'rrx'), 'or': (7, 'rrx'), 'not': (8, 'rr'), 'xor': (9, 'rrx'),
    'jmp': (10, 'j'), 'jn': (11, 'j'), 'jz': (12, 'j'), 'jp': (13, 'j'),
    'load': (14, 'rs'), 'load2': (15, 'rs'), 'stor': (16, 'rs'), 'stor2': (17, 'rs'),
    'in': (18, 'rs'), 'out': (19, 'rs'), 'push': (20, 'rs'), 'pop': (21, 'rr'),
    'move': (22, 'rs'), 'swap': (23, 'rr'), 'call': (24, 'rs'), 'ret': (25, 'r'),
    'id': (26, 'rs'), 'shl': (27, 'rrx'), 'shr': (28, 'rrx'), 'sar': (29, 'rrx'),
    'div': (30, 'rrx'), 'mod': (31, 'rrx'), 'cmp': (32, 'cmp'),
    'jnz': (33, 'j'), 'jle': (34, 'j'), 'jge': (35, 'j'),
    'sdiv': (36, 'rrx'), 'smod': (37, 'rrx'), 'adc': (38, 'rrx'),
    'sbb': (39, 'rrx'), 'jc': (40, 'j')
}
M2_ALIAS = {'mov': 'move', 'li': 'move'}
M2_MACROS = {'inc', 'dec', 'neg', 'clr', 'tst'}


# ALU instructions accept [reg] memory operands: [dst] -> bit 5, [src1] -> bit 4,
# [src2] -> bit 3 of the first word.  Roles by operand position:
MEM_BIT = {'d': 0x20, '1': 0x10, '2': 0x08}
MEM_ROLES = {'rrx': 'd12', 'cmp': '12'}


REG_OFF_RE = re.compile(r'^\s*(r\d+|lr)\s*([+-])\s*(.+?)\s*$', re.I)


def split_reg_off(tok, maxreg, env):
    """Split `rN +/- expr` into (reg, signed_offset). None if not reg-based."""
    m = REG_OFF_RE.match(tok)
    if not m:
        return None
    r = as_reg(m.group(1), maxreg)
    if r is None:
        return None
    v = eval_expr('%s(%s)' % (m.group(2), m.group(3)), env)
    return (r, v)


def enc2(env, mn, ops, cc):
    code, shape = M2[mn]
    R = lambda t: reg(t, 15)
    w0 = code << 8 | (0x80 if cc else 0)
    dr = o1 = o2 = 0
    soff = doff = None      # signed offsets for or1 (low) / dr (high)

    def set_soff(v):
        nonlocal soff
        if soff is not None:
            raise AsmError("duplicate source offset")
        soff = v

    def set_doff(v):
        nonlocal doff
        if doff is not None:
            raise AsmError("duplicate destination offset")
        doff = v

    def no_off(tok, what):
        if split_reg_off(tok, 15, env) is not None:
            raise AsmError("offsets not allowed on %s" % what)
        return R(tok)

    if shape == 'rrx' and len(ops) == 2:        # `add rd, x`  ==  `add rd, rd, x`
        ops = [ops[0], ops[0], ops[1]]

    roles = MEM_ROLES.get(shape, 'd1' if mn == 'not' else '')
    bases, mems = [], []
    for i, o in enumerate(ops):
        t = o.strip()
        is_mem = False
        if t.startswith('['):
            if not t.endswith(']'):
                raise AsmError("missing ']'")
            if i >= len(roles):
                raise AsmError("'%s' has no memory-operand form for operand %d" % (mn, i + 1))
            w0 |= MEM_BIT[roles[i]]
            t = t[1:-1].strip()
            is_mem = True
        bases.append(t)
        mems.append(is_mem)
    ops = bases

    extra = []           # the optional third word (16-bit immediate, flag i3)

    def imm16(tok):
        nonlocal w0
        v = eval_expr(tok, env)
        if env.final and not -32768 <= v <= 65535:
            raise AsmError("immediate %d does not fit in 16 bits" % v)
        w0 |= 0x04
        extra.append(v & 0xFFFF)

    def reg_or_imm(tok):
        """ALU second operand: register, 4-bit immediate (i2) or 16-bit immediate (i3).
        i2 is only chosen when the value is known in pass 1 and fits 0..15, so a
        line's size never depends on a forward reference."""
        nonlocal w0
        r = as_reg(tok, 15)
        if r is not None:
            return r
        env.undef = False
        v = eval_expr(tok, env)
        it = env.item
        if not env.final:
            it.i3 = env.undef or not 0 <= v <= 15
        if it.i3:
            imm16(tok)
            return 0
        w0 |= 0x40
        return chk_unsigned(env, v, 4, "immediate")

    if shape == 'none':
        need(mn, ops, 0)
    elif shape == 'r':                        # ret: dr is SP, +off = post-adjust
        need(mn, ops, 1)
        ro = split_reg_off(ops[0], 15, env)
        dr = ro[0] if ro is not None else R(ops[0])
        if ro is not None:
            set_doff(ro[1])
    elif shape == 'j':                        # jump target: register or 16-bit address
        need(mn, ops, 1)                      # `jmp r1+5` uses high byte; imm folds
        ro = split_reg_off(ops[0], 15, env)
        if ro is not None:
            dr = ro[0]
            set_doff(ro[1])
        else:
            r = as_reg(ops[0], 15)
            if r is not None:
                dr = r
            else:
                imm16(ops[0])
    elif shape == 'rr':
        need(mn, ops, 2)
        if mn == 'swap':                      # both sides take offsets
            ro0 = split_reg_off(ops[0], 15, env)
            ro1 = split_reg_off(ops[1], 15, env)
            dr = ro0[0] if ro0 is not None else R(ops[0])
            o1 = ro1[0] if ro1 is not None else R(ops[1])
            if ro0 is not None:
                set_doff(ro0[1])
            if ro1 is not None:
                set_soff(ro1[1])
        elif mn == 'pop':                     # `pop rd, rsp+off` peeks ahead
            dr = no_off(ops[0], "destination")
            ro = split_reg_off(ops[1], 15, env)
            if ro is not None:
                o1 = ro[0]
                set_soff(ro[1])
            else:
                o1 = R(ops[1])
        else:                                 # not: `not rd, rs+off`, `[rd+off]`
            if mems[0]:
                ro = split_reg_off(ops[0], 15, env)
                if ro is not None:
                    dr = ro[0]
                    set_doff(ro[1])
                else:
                    dr = R(ops[0])
            else:
                dr = no_off(ops[0], "destination")
            ro = split_reg_off(ops[1], 15, env)
            if ro is not None:
                o1 = ro[0]
                set_soff(ro[1])
            else:
                o1 = R(ops[1])
    elif shape == 'rs':                     # second operand: register or 16-bit immediate
        need(mn, ops, 2)
        if mn in ('stor', 'stor2', 'out'):
            ro = split_reg_off(ops[0], 15, env)   # address/port offset (high)
            if ro is not None:
                dr = ro[0]
                set_doff(ro[1])
            else:
                dr = R(ops[0])
        else:                                 # dest is a plain register
            dr = no_off(ops[0], "destination")
        ro = split_reg_off(ops[1], 15, env)       # value/port/target offset (low)
        if ro is not None:
            o1 = ro[0]
            set_soff(ro[1])
        else:
            r = as_reg(ops[1], 15)
            if r is not None:
                o1 = r
            else:
                imm16(ops[1])
    elif shape == 'rrx':
        need(mn, ops, 3)
        if mems[0]:                           # `[rd+off]` destination offset
            ro = split_reg_off(ops[0], 15, env)
            if ro is not None:
                dr = ro[0]
                set_doff(ro[1])
            else:
                dr = R(ops[0])
        else:
            dr = no_off(ops[0], "destination")
        ro = split_reg_off(ops[1], 15, env)       # src1 value or [mem] offset
        if ro is not None:
            o1 = ro[0]
            set_soff(ro[1])
        else:
            o1 = R(ops[1])
        if split_reg_off(ops[2], 15, env) is not None:
            raise AsmError("offsets not allowed on the third operand")
        o2 = reg_or_imm(ops[2])
    elif shape == 'cmp':
        need(mn, ops, 2)
        ro = split_reg_off(ops[0], 15, env)
        if ro is not None:
            o1 = ro[0]
            set_soff(ro[1])
        else:
            o1 = R(ops[0])
        if split_reg_off(ops[1], 15, env) is not None:
            raise AsmError("offsets not allowed on the second operand")
        o2 = reg_or_imm(ops[1])
    if soff is not None or doff is not None:
        s, d = soff or 0, doff or 0
        if env.final:
            for v, what in ((s, "source"), (d, "destination")):
                if not -255 <= v <= 255:
                    raise AsmError("%s offset %d out of range (-255..255)" % (what, v))
        if (s > 0 or d > 0) and (s < 0 or d < 0):
            raise AsmError("mixed + and - offsets need the same sign (shared i5)")
        neg = s < 0 or d < 0
        w0 |= 0x02 | (0x01 if neg else 0)
        extra.append(((abs(d) << 8) | abs(s)) & 0xFFFF)
    return [w0, dr << 12 | o1 << 8 | o2 << 4] + extra


def macro2(env, mn, ops, cc):
    """Mode-2 macros.  `.cc` applies to the instruction that produces the result.
    Bracketed [reg] memory operands work too: inc [r1], neg [r2], clr [r3]."""
    if mn in ('inc', 'dec'):                    # inc rd | inc rd, rs
        if len(ops) not in (1, 2):
            raise AsmError("'%s' takes 1 or 2 operands" % mn)
        src = ops[1] if len(ops) == 2 else ops[0]
        return enc2(env, 'add' if mn == 'inc' else 'sub', [ops[0], src, '1'], cc)
    if mn == 'neg':                             # neg rd | neg rd, rs   (not; add 1)
        if len(ops) not in (1, 2):
            raise AsmError("'neg' takes 1 or 2 operands")
        src = ops[1] if len(ops) == 2 else ops[0]
        return (enc2(env, 'not', [ops[0], src], False) +
                enc2(env, 'add', [ops[0], ops[0], '1'], cc))
    if mn == 'clr':                             # clr rd  (rd = rd ^ rd)
        need(mn, ops, 1)
        return enc2(env, 'xor', [ops[0]] * 3, cc)
    if mn == 'tst':                             # tst rd  (cmp rd, 0)
        need(mn, ops, 1)
        return enc2(env, 'cmp', [ops[0], '0'], False)
    raise AsmError("internal: unhandled mode-2 macro '%s'" % mn)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
class Item:
    def __init__(self, lineno, text, labels, mn, optext):
        self.lineno, self.text, self.labels, self.mn, self.optext = lineno, text, labels, mn, optext
        self.i3 = None        # mode 2: does this line need the 3rd (16-bit immediate) word?


def strip_comment(line):
    out, q, i = [], None, 0
    while i < len(line):
        c = line[i]
        if q:
            out.append(c)
            if c == '\\' and i + 1 < len(line):
                out.append(line[i + 1])
                i += 2
                continue
            if c == q:
                q = None
        elif c in '"\'':
            q = c
            out.append(c)
        elif c == ';' or line.startswith('//', i):
            break
        else:
            out.append(c)
        i += 1
    return ''.join(out)


def split_ops(s):
    parts, cur, depth, q, i = [], [], 0, None, 0
    while i < len(s):
        c = s[i]
        if q:
            cur.append(c)
            if c == '\\' and i + 1 < len(s):
                cur.append(s[i + 1])
                i += 2
                continue
            if c == q:
                q = None
        elif c in '"\'':
            q = c
            cur.append(c)
        elif c in '([':
            depth += 1
            cur.append(c)
        elif c in ')]':
            depth -= 1
            cur.append(c)
        elif c == ',' and depth == 0:
            parts.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    last = ''.join(cur).strip()
    if last or parts:
        parts.append(last)
    return parts


def decode_string(tok):
    tok = tok.strip()
    if len(tok) < 2 or tok[0] != '"' or tok[-1] != '"':
        raise AsmError("expected a \"quoted string\"")
    body, out, i = tok[1:-1], [], 0
    while i < len(body):
        c = body[i]
        if c == '\\':
            i += 1
            if i >= len(body):
                raise AsmError("dangling backslash in string")
            e = body[i]
            if e == 'x':
                out.append(int(body[i + 1:i + 3], 16))
                i += 2
            elif e in ESC:
                out.append(ESC[e])
            else:
                raise AsmError("unknown escape '\\%s'" % e)
        else:
            out.append(ord(c))
        i += 1
    return out


LABEL_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*:(?!=)(.*)$')
EQ_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*(.+)$')


def parse_source(text, errors):
    items = []
    for no, raw in enumerate(text.splitlines(), 1):
        line = strip_comment(raw).strip()
        labels = []
        while True:
            m = LABEL_RE.match(line)
            if not m:
                break
            labels.append(m.group(1))
            line = m.group(2).strip()
        if not line:
            if labels:
                items.append(Item(no, raw, labels, None, ''))
            continue
        m = EQ_RE.match(line)
        if m and not line.startswith('.'):
            items.append(Item(no, raw, labels, '.equ', '%s, %s' % (m.group(1), m.group(2))))
            continue
        m = re.match(r'(\S+)\s*(.*)$', line)
        items.append(Item(no, raw, labels, m.group(1).lower(), m.group(2)))
    return items


# --------------------------------------------------------------------------
# directives and instruction dispatch
# --------------------------------------------------------------------------
def word_val(env, v):
    if env.final and not -32768 <= v <= 65535:
        raise AsmError("value %d does not fit in 16 bits" % v)
    return v & 0xFFFF


def directive(env, it):
    d, ops = it.mn, split_ops(it.optext)
    if d == '.org':
        need(d, ops, 1)
        a = eval_expr(ops[0], env, strict=True)
        if not 0 <= a <= 0xFFFF:
            raise AsmError(".org address out of range")
        env.pc = a
        return []
    if d in ('.mode', '.mode1', '.mode2'):
        if d == '.mode':
            need(d, ops, 1)
            n = ops[0].strip()
        else:
            n = d[-1]
        if n not in ('1', '2'):
            raise AsmError(".mode takes 1 or 2")
        env.mode = int(n)
        return []
    if d == '.equ':
        need(d, ops, 2)
        name = ops[0]
        if not re.match(r'^[A-Za-z_]\w*$', name):
            raise AsmError("bad symbol name '%s'" % name)
        if not env.final:
            if name in env.syms:
                raise AsmError("symbol '%s' already defined" % name)
            env.syms[name] = eval_expr(ops[1], env, strict=True)
        return []
    if d == '.word':
        if not ops:
            raise AsmError(".word needs at least one value")
        return [word_val(env, eval_expr(o, env)) for o in ops]
    if d in ('.string', '.asciz', '.ascii'):
        if not ops:
            raise AsmError("%s needs a string" % d)
        out = []
        for o in ops:
            if o.startswith('"'):
                out += decode_string(o)
            else:
                out.append(word_val(env, eval_expr(o, env)))
        if d != '.ascii':
            out.append(0)
        return out
    if d in ('.fill', '.space', '.blkw'):
        if not 1 <= len(ops) <= 2:
            raise AsmError("%s takes: count [, value]" % d)
        n = eval_expr(ops[0], env, strict=True)
        if n < 0:
            raise AsmError("negative count")
        v = word_val(env, eval_expr(ops[1], env)) if len(ops) == 2 else 0
        return [v] * n
    raise AsmError("unknown directive '%s'" % d)


def process(env, it):
    mn = it.mn
    env.item = it
    if mn.startswith('.'):
        return directive(env, it)
    ops = split_ops(it.optext)
    base, _, suf = mn.partition('.')
    if suf and suf != 'cc':
        raise AsmError("unknown suffix '.%s' (only .cc exists)" % suf)
    if env.mode == 1:
        if suf:
            raise AsmError(".cc is only for mode-2 instructions")
        if base in M1_NAMES or BR_RE.match(base):
            return enc1(env, base, ops)
        if M2_ALIAS.get(base, base) in M2:
            raise AsmError("'%s' is a mode-2 instruction; switch with `.mode 2`" % base)
    else:
        if base in M2_MACROS:
            return macro2(env, base, ops, bool(suf))
        b = M2_ALIAS.get(base, base)
        if b in M2:
            return enc2(env, b, ops, bool(suf))
        if base in M1_NAMES or BR_RE.match(base):
            raise AsmError("'%s' is a mode-1 instruction; switch with `.mode 1`" % base)
    raise AsmError("unknown instruction '%s'" % mn)


def run_pass(items, syms, final, errors, mem=None, listing=None):
    env = Env(syms, final)
    for it in items:
        try:
            nonemit = it.mn in ('.org', '.mode', '.mode1', '.mode2')
            words = []
            if it.mn is not None and nonemit:
                process(env, it)
            if not final:
                for l in it.labels:
                    if l in syms:
                        raise AsmError("symbol '%s' already defined" % l)
                    syms[l] = env.pc
            if it.mn is not None and not nonemit:
                words = process(env, it)
            if final and words:
                if env.pc + len(words) > 0x10000:
                    raise AsmError("program does not fit in 64K words")
                for i, w in enumerate(words):
                    if env.pc + i in mem:
                        raise AsmError("address 0x%04X is already occupied (overlap)" % (env.pc + i))
                    mem[env.pc + i] = w
            if final and listing is not None:
                listing.append((env.pc, words, it))
            env.pc += len(words)
        except AsmError as e:
            errors.append((it.lineno, str(e)))
    return env


# predefined symbols, for `id` selectors and feature bits (keep in sync with bob16.c)
BUILTINS = {
    'ID_MAX': 0, 'ID_VERSION': 1, 'ID_FEATURES': 2, 'ID_MEMTOP': 3, 'ID_RS0': 4,
    'ID_REGS': 5, 'ID_OPCODES': 6, 'ID_PORTS': 7,
    'ID_NAME0': 8, 'ID_NAME1': 9, 'ID_NAME2': 10,
    'FEAT_MEMOPS': 1, 'FEAT_IMM16': 2, 'FEAT_SDIV': 4, 'FEAT_IO': 8, 'FEAT_STACK': 16,
    'FEAT_OFF': 32,
}


def assemble(text):
    errors, syms, mem, listing = [], dict(BUILTINS), {}, []
    items = parse_source(text, errors)
    run_pass(items, syms, False, errors)
    if errors:
        return None, None, errors
    run_pass(items, syms, True, errors, mem, listing)
    if errors:
        return None, None, errors
    return mem, listing, errors


def main():
    ap = argparse.ArgumentParser(description="BOB-16 assembler")
    ap.add_argument("source")
    ap.add_argument("-o", "--output", help="output .bin (default: source with .bin)")
    ap.add_argument("-l", "--listing", help="write a listing file")
    ap.add_argument("--big-endian", action="store_true", help="emit big-endian words")
    a = ap.parse_args()

    try:
        with open(a.source) as f:
            text = f.read()
    except OSError as e:
        sys.exit("bob16asm: %s" % e)

    mem, listing, errors = assemble(text)
    if errors:
        for no, msg in sorted(errors):
            print("%s:%d: error: %s" % (a.source, no, msg), file=sys.stderr)
        sys.exit(1)

    n = (max(mem) + 1) if mem else 0
    data = bytearray()
    for addr in range(n):
        w = mem.get(addr, 0)
        data += bytes((w >> 8, w & 0xFF)) if a.big_endian else bytes((w & 0xFF, w >> 8))
    out = a.output or re.sub(r'\.[^./\\]*$', '', a.source) + '.bin'
    with open(out, 'wb') as f:
        f.write(data)

    if a.listing:
        with open(a.listing, 'w') as f:
            for addr, words, it in listing:
                if not words:
                    f.write("%-14s  %s\n" % ("", it.text.rstrip()))
                    continue
                for i in range(0, len(words), 2):
                    chunk = ' '.join('%04X' % w for w in words[i:i + 2])
                    f.write("%04X  %-9s %s\n" % (addr + i, chunk, it.text.rstrip() if i == 0 else ''))
    print("%s: %d words -> %s" % (a.source, n, out))


if __name__ == '__main__':
    main()
