#!/usr/bin/env python3
"""
bob16conv.py - convert assembly for the ORIGINAL bob16
(https://github.com/somerandomviolinkid/bob16) into assembly for this fork's
assembler (bob16asm.py).

Usage:  bob16conv.py prog.basm [-o prog.asm] [-a] [-q]

  -o FILE         output file (default: input name with .asm; '-' = stdout)
  -a              annotate every converted line with the original source
  -q              don't print warnings

What the original accepts (and therefore what this reads):
  * one instruction per line, operands separated by whitespace
  * ';' comment lines (inline comments and commas are tolerated here)
  * no labels: ld/ldi/st/sti/lea/br/jsr take a decimal offset relative to the
    *incremented* pc.  They become labels (L0004, ...) in the output.
  * .fill HEX   (hexadecimal, with or without 0x)   ->  .word 0xHEX
  * .stringz TEXT  (one token, no spaces)           ->  .string "TEXT"

The original assembler is buggy in a few places (in-place add/and, add with an
immediate, jsr, jsrr, nop).  This converter follows the DOCUMENTED semantics
and prints a warning on every line where the original assembler would have
produced something different.

Every instruction converts to exactly one word, so addresses are preserved and
literal addresses (e.g. a `.fill` holding a pointer) stay valid.
"""
import argparse
import re
import sys


class ConvError(Exception):
    pass


# --------------------------------------------------------------------------
# parsing helpers (mirroring the original assembler's quirks)
# --------------------------------------------------------------------------
REG_RE = re.compile(r'^[rR](\d+)$')
DEC_RE = re.compile(r'^[+-]?\d+$')
TRAP_NAMES = {0: 'halt', 1: 'putc', 2: 'puts', 3: 'gets'}


def atoi(tok):
    """C atoi(): optional sign + leading digits, anything else -> 0."""
    m = re.match(r'\s*([+-]?\d+)', tok)
    return int(m.group(1)) if m else 0


def parse_reg(tok):
    m = REG_RE.match(tok)
    if not m:
        return None
    n = int(m.group(1))
    if n > 7:
        raise ConvError("register %s doesn't exist (r0-r7)" % tok)
    return n


def need_reg(tok):
    r = parse_reg(tok)
    if r is None:
        raise ConvError("expected a register, got '%s'" % tok)
    return r


def esc_string(s):
    out = []
    for ch in s:
        c = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == '\\':
            out.append('\\\\')
        elif 32 <= c < 127:
            out.append(ch)
        else:
            out.append('\\x%02x' % (c & 0xFF))
    return ''.join(out)


def label(addr):
    return 'L%04X' % addr


class Stmt:
    def __init__(self, kind, lineno, raw, toks=None, comment=''):
        self.kind = kind          # blank | comment | instr | fill | stringz
        self.lineno = lineno
        self.raw = raw
        self.toks = toks or []
        self.comment = comment
        self.addr = 0             # original word address
        self.size = 0
        self.lines = []           # converted output lines (with @T0@ placeholders)
        self.targets = []         # original addresses the placeholders refer to


# --------------------------------------------------------------------------
# per-instruction conversion
# --------------------------------------------------------------------------
class Converter:
    def __init__(self):
        self.warnings = []        # (lineno, text)
        self.cur = None

    def warn(self, msg):
        self.warnings.append((self.cur.lineno, msg))

    def imm(self, tok, lo, hi):
        if not DEC_RE.match(tok):
            self.warn("'%s' is not a decimal integer; the original reads it as %d"
                      % (tok, atoi(tok)))
        v = atoi(tok)
        if v < lo or v > hi:
            c = max(lo, min(hi, v))
            self.warn("immediate %d is out of range %d..%d; the original clamps it to %d"
                      % (v, lo, hi, c))
            v = c
        return v

    def rel_target(self, tok, lo, hi):
        off = self.imm(tok, lo, hi)
        return off, (self.cur.addr + 1 + off) & 0xFFFF

    # ---- instruction table
    def convert(self, st):
        self.cur = st
        t = st.toks
        mn = t[0].lower()
        n = len(t)

        def count(*allowed):
            if n not in allowed:
                raise ConvError("'%s' takes %s operand(s), got %d"
                                % (mn, ' or '.join(str(a - 1) for a in allowed), n - 1))

        if mn == 'nop':
            count(1)
            self.warn("the original assembler doesn't recognise 'nop'")
            return ['nop'], []

        if mn in ('add', 'and'):
            count(3, 4)
            rd = need_reg(t[1])
            if n == 4:
                rs1 = need_reg(t[2])
                r3 = parse_reg(t[3])
                if r3 is not None:
                    return ['%s r%d, r%d, r%d' % (mn, rd, rs1, r3)], []
                if mn == 'add':
                    self.warn("the original assembler rejects 'add rd rs imm' "
                              "(\"Wrong tokens\"); converted as documented")
                return ['%s r%d, r%d, %d' % (mn, rd, rs1, self.imm(t[3], -8, 7))], []
            self.warn("the original assembler mis-encodes in-place '%s'; "
                      "converted as documented" % mn)
            rs = parse_reg(t[2])
            if rs is not None:
                return ['%s r%d, r%d' % (mn, rd, rs)], []
            return ['%s r%d, %d' % (mn, rd, self.imm(t[2], -64, 63))], []

        if mn == 'not':
            count(2, 3)
            rd = need_reg(t[1])
            if n == 3:
                return ['not r%d, r%d' % (rd, need_reg(t[2]))], []
            return ['not r%d' % rd], []

        if mn in ('ld', 'ldi', 'st', 'sti', 'lea'):
            if mn == 'lea':
                count(2, 3)
            else:
                count(3)
            rd = need_reg(t[1])
            if n == 2:                      # README form `lea r0`
                off, tgt = 0, (st.addr + 1) & 0xFFFF
            else:
                off, tgt = self.rel_target(t[2], -256, 255)
            return ['%-5s r%d, @T0@' % (mn, rd)], [tgt]

        if mn in ('ldr', 'str'):
            count(4)
            rd, rb = need_reg(t[1]), need_reg(t[2])
            return ['%s r%d, r%d, %d' % (mn, rd, rb, self.imm(t[3], -32, 31))], []

        if mn == 'br':
            count(3)
            flags = t[1]
            if not 1 <= len(flags) <= 3:
                raise ConvError("br condition must be 1-3 letters from n, z, p")
            fl = set()
            for ch in flags:
                if ch.lower() in 'nzp':
                    fl.add(ch.lower())
                else:
                    self.warn("br flag '%s' is ignored by the original assembler" % ch)
            if any(c.isupper() and c.lower() in 'nzp' for c in flags):
                self.warn("the original only recognises lowercase n/z/p; "
                          "uppercase flags are read as intended here")
            off, tgt = self.rel_target(t[2], -256, 255)
            if not fl:
                self.warn("no valid condition flags: this branch never happens, so it becomes nop")
                return ['nop'], []
            return ['br%s @T0@' % ''.join(c for c in 'nzp' if c in fl)], [tgt]

        if mn == 'jmp':
            count(2)
            return ['jmp r%d' % need_reg(t[1])], []

        if mn == 'jsr':
            count(2)
            self.warn("the original assembler sets the jsrr bit on 'jsr', so the "
                      "original emulator would not run it as written")
            off, tgt = self.rel_target(t[1], -1024, 1023)
            return ['jsr   @T0@'], [tgt]

        if mn == 'jsrr':
            count(2)
            self.warn("the original assembler cannot assemble 'jsrr' (it always "
                      "reports \"Wrong tokens\"); converted as documented")
            return ['jsrr  r%d' % need_reg(t[1])], []

        if mn == 'ret':
            count(1)
            return ['ret'], []

        if mn == 'trap':
            count(2)
            v = atoi(t[1])
            if not DEC_RE.match(t[1]):
                self.warn("'%s' is not a decimal integer; the original reads it as %d"
                          % (t[1], v))
            if not 0 <= v <= 15:
                raise ConvError("bad trap vector %d (0-15)" % v)
            if v in TRAP_NAMES:
                return [TRAP_NAMES[v]], []
            self.warn("trap %d does nothing in the original; becomes nop here (in the "
                      "fork it would not be a no-op)" % v)
            return ['nop    ; trap %d' % v], []

        raise ConvError("unknown opcode '%s'" % t[0])


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def parse_hex16(tok):
    m = re.match(r'\s*([+-]?)(?:0[xX])?([0-9a-fA-F]+)', tok)
    if not m:
        raise ConvError("bad .fill value '%s' (hexadecimal expected)" % tok)
    v = int(m.group(2), 16)
    if m.group(1) == '-':
        v = -v
    return v & 0xFFFF


def split_source(text):
    stmts = []
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            stmts.append(Stmt('blank', no, raw))
            continue
        if line.startswith(';'):
            stmts.append(Stmt('comment', no, raw, comment=line))
            continue
        code, sep, com = line.partition(';')
        toks = code.replace(',', ' ').split()
        if not toks:
            stmts.append(Stmt('comment', no, raw, comment=line))
            continue
        k = toks[0].lower()
        kind = 'fill' if k == '.fill' else 'stringz' if k == '.stringz' else 'instr'
        stmts.append(Stmt(kind, no, raw, toks, (sep + com) if sep else ''))
    return stmts


def convert_program(text):
    stmts = split_source(text)
    conv = Converter()
    errors = []

    # ---- pass 1: addresses
    addr = 0
    for st in stmts:
        st.addr = addr
        if st.kind in ('instr', 'fill'):
            st.size = 1
        elif st.kind == 'stringz':
            if len(st.toks) != 2:
                errors.append((st.lineno, "'.stringz' takes exactly one token (no spaces)"))
                st.size = 1
            else:
                st.size = len(st.toks[1]) + 1
        addr += st.size
    total = addr
    if total > 0xFFFF:
        errors.append((0, "program too large (%d words)" % total))

    # ---- pass 2: convert statements
    targets = set()
    for st in stmts:
        try:
            if st.kind == 'fill':
                if len(st.toks) != 2:
                    raise ConvError("'.fill' takes exactly one value")
                st.lines = ['.word 0x%04X' % parse_hex16(st.toks[1])]
            elif st.kind == 'instr':
                st.lines, st.targets = conv.convert(st)
                targets.update(st.targets)
        except ConvError as e:
            errors.append((st.lineno, str(e)))
    if errors:
        return None, conv, errors

    return (stmts, targets, total), conv, errors


def render(stmts, targets, total, annotate, name, conv):
    out = ['; converted from %s by bob16conv.py' % name]
    outside = sorted(t for t in targets if t > total)
    for t in outside:
        out.append('.equ  %s, 0x%04X    ; beyond the end of the program' % (label(t), t))
    if outside:
        out.append('')

    def emit_label(a):
        if a in targets:
            out.append('%s:' % label(a))

    for st in stmts:
        if st.kind == 'blank':
            out.append('')
            continue
        if st.kind == 'comment':
            out.append(st.comment)
            continue

        if st.kind == 'stringz':
            s = st.toks[1]
            n = len(s)
            cuts = sorted(t - st.addr for t in targets if st.addr < t <= st.addr + n)
            bounds = [0] + cuts + [n]
            for i in range(len(bounds) - 1):
                a, b = bounds[i], bounds[i + 1]
                emit_label(st.addr + a)
                last = i == len(bounds) - 2
                if last:                      # last piece carries the terminator
                    out.append('        .string "%s"' % esc_string(s[a:b]))
                elif b > a:
                    out.append('        .ascii  "%s"' % esc_string(s[a:b]))
            if annotate:
                out[-1] = '%-36s ; orig: %s' % (out[-1], ' '.join(st.toks))
            continue

        emit_label(st.addr)
        lines = [ln.replace('@T0@', label(st.targets[0])) if st.targets else ln
                 for ln in st.lines]
        for i, ln in enumerate(lines):
            m = re.match(r'(\S+)\s*(.*)$', ln)
            text = '        %-5s %s' % (m.group(1), m.group(2))
            text = text.rstrip()
            if annotate and i == 0:
                text = '%-36s ; orig: %s' % (text, ' '.join(st.toks))
            out.append(text)
    emit_label(total)
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser(description="convert original bob16 assembly to this fork's assembly")
    ap.add_argument('source')
    ap.add_argument('-o', '--output')
    ap.add_argument('-a', '--annotate', action='store_true')
    ap.add_argument('-q', '--quiet', action='store_true')
    a = ap.parse_args()

    try:
        with open(a.source, 'rb') as f:
            text = f.read().decode('latin-1')
    except OSError as e:
        sys.exit('bob16conv: %s' % e)

    res, conv, errors = convert_program(text)
    if errors:
        for no, msg in sorted(errors):
            print('%s:%d: error: %s' % (a.source, no, msg), file=sys.stderr)
        sys.exit(1)

    stmts, targets, total = res
    result = render(stmts, targets, total, a.annotate, a.source, conv)

    if not a.quiet:
        for no, msg in conv.warnings:
            print('%s:%d: warning: %s' % (a.source, no, msg), file=sys.stderr)

    if a.output == '-':
        sys.stdout.write(result)
    else:
        out = a.output or re.sub(r'\.[^./\\]*$', '', a.source) + '.asm'
        if out == a.source:
            out += '.asm'
        with open(out, 'wb') as f:
            f.write(result.encode('latin-1', 'replace'))
        print('%s -> %s' % (a.source, out))


if __name__ == '__main__':
    main()
