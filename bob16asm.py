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
  jmp rd | ret | jsr label | jsrr rs
  trap n | halt | putc | puts | gets | ext
  Pseudo: mov rd, rs | clr rd | li rd, imm(-64..63) | enter2 label

MODE 2 (32-bit instructions, entered with `trap ext` with r7 = code address;
        left again with `uext rX`)
  Add `.cc` to a mnemonic to update the condition codes, e.g. add.cc
  nop, halt
  uext rd
  add sub mul shl shr sar div mod sdiv smod   rd, rs1, rs2|imm4(0..15)
  and or xor                                  rd, rs1, rs2
  not rd, rs      move rd, rs      swap rd, rs     id rd, rs
  load rd, raddr  load2 rd, raddr  stor raddr, rval   stor2 raddr, rval
  in rd, rport    out rport, rval
  push rsp, rval  pop rd, rsp      call rsp, rtarget  ret rsp
  cmp rs1, rs2|imm4
  jmp jn jz jp jnz jle jge  rtarget
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
            'trap', 'mov', 'clr', 'li', 'enter2'} | set(PCREL) | set(TRAPS)
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

    if mn == 'jsr':
        need(mn, ops, 1)
        off = rel_off(env, eval_expr(ops[0], env))
        if env.final and not 0 <= off <= 1023:
            raise AsmError(
                "jsr can only reach 0..+1023 words forward (distance is %d). The emulator "
                "treats bit 10 of the offset as the jsr/jsrr selector, so backward calls "
                "can't be encoded; use  lea r6, target ; jsrr r6  instead" % off)
        return [12 << 12 | (off & 0x3FF)]

    if mn == 'jsrr':
        need(mn, ops, 1)
        r = R(ops[0])
        if r < 4:
            raise AsmError("jsrr can only use r4..r7 (the emulator reads the register "
                           "from bits 10:8 and bit 10 is always set)")
        return [12 << 12 | r << 8]

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

    raise AsmError("internal: unhandled mode-1 mnemonic '%s'" % mn)


# --------------------------------------------------------------------------
# mode 2
# --------------------------------------------------------------------------
M2 = {
    'nop': (0, 'none'), 'halt': (1, 'none'), 'uext': (2, 'r'),
    'add': (3, 'rrx'), 'sub': (4, 'rrx'), 'mul': (5, 'rrx'),
    'and': (6, 'rrr'), 'or': (7, 'rrr'), 'not': (8, 'rr'), 'xor': (9, 'rrr'),
    'jmp': (10, 'r'), 'jn': (11, 'r'), 'jz': (12, 'r'), 'jp': (13, 'r'),
    'load': (14, 'rr'), 'load2': (15, 'rr'), 'stor': (16, 'rr'), 'stor2': (17, 'rr'),
    'in': (18, 'rr'), 'out': (19, 'rr'), 'push': (20, 'rr'), 'pop': (21, 'rr'),
    'move': (22, 'rr'), 'swap': (23, 'rr'), 'call': (24, 'rr'), 'ret': (25, 'r'),
    'id': (26, 'rr'), 'shl': (27, 'rrx'), 'shr': (28, 'rrx'), 'sar': (29, 'rrx'),
    'div': (30, 'rrx'), 'mod': (31, 'rrx'), 'cmp': (32, 'cmp'),
    'jnz': (33, 'r'), 'jle': (34, 'r'), 'jge': (35, 'r'),
    'sdiv': (36, 'rrx'), 'smod': (37, 'rrx'),
}
M2_ALIAS = {'mov': 'move'}


def enc2(env, mn, ops, cc):
    code, shape = M2[mn]
    R = lambda t: reg(t, 15)
    w0 = code << 8 | (0x80 if cc else 0)
    dr = o1 = o2 = 0

    def reg_or_imm4(tok):
        nonlocal w0
        r = as_reg(tok, 15)
        if r is not None:
            return r
        w0 |= 0x40
        return chk_unsigned(env, eval_expr(tok, env), 4, "immediate")

    if shape == 'none':
        need(mn, ops, 0)
    elif shape == 'r':
        need(mn, ops, 1)
        dr = R(ops[0])
    elif shape == 'rr':
        need(mn, ops, 2)
        dr, o1 = R(ops[0]), R(ops[1])
    elif shape == 'rrr':
        need(mn, ops, 3)
        dr, o1, o2 = R(ops[0]), R(ops[1]), R(ops[2])
    elif shape == 'rrx':
        need(mn, ops, 3)
        dr, o1 = R(ops[0]), R(ops[1])
        o2 = reg_or_imm4(ops[2])
    elif shape == 'cmp':
        need(mn, ops, 2)
        o1 = R(ops[0])
        o2 = reg_or_imm4(ops[1])
    return [w0, dr << 12 | o1 << 8 | o2 << 4]


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
class Item:
    def __init__(self, lineno, text, labels, mn, optext):
        self.lineno, self.text, self.labels, self.mn, self.optext = lineno, text, labels, mn, optext


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


def assemble(text):
    errors, syms, mem, listing = [], {}, {}, []
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

