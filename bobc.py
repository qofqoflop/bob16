#!/usr/bin/env python3
"""
bobc.py - M2 compiler for a C-like language targeting BOB-16 (v1.4).

Usage:  bobc.py prog.b [-o prog.asm]

M2 subset:
  types:        int | unsigned | char | void (return type only; all one word)
  top level:    globals + functions; exactly one `int main()` / `int main(void)`
  globals:      int g;  int h = 5;  int a[10];  char *s = "hi";
  functions:    int add(int a, int b) { return a + b; }  void f(void) {...}
                params (arrays decay to pointers), calls, recursion
  locals:       same as globals, inside any function (arrays included)
  statements:   { }  if/else  while  do-while  for  return  break  continue
  expressions:  full C precedence, = += ... ?:  && || ! ~ - * & [] () calls,
                sizeof, numbers, 'c', "str" (builtins take strings)
  builtins (statements only): puts(e)  putc(e)  print_int(e)
  gettime(e)/settime(e) (64-bit virtual time <-> four words at e)

M3 adds: `half` (IEEE binary16) variables/globals/params/returns, `h`
  literals (`1.5h 2h infh nanh`), mixed int/half arithmetic (int promotes
  via `itof`), `fdiv` (never traps: x/0 = Inf), `ftoi` on half->int
  conversion (traps out of range and on NaN). No `print_half` yet: assign
  to an int and use print_int.

Structs+enums: `struct P { int x; half y; };` (top level; members may be
  scalars, pointers, arrays, nested structs), `p.x` / `p->x` / `a[i].x`,
  struct assignment copies, `&s` / `&s.x`; struct params/returns must be
  pointers. `enum C { RED, GREEN = 5, BLUE };` gives int constants;
  `enum C` is an int-typed variable. No initializers inside structs.

Methods+unions: `void P.move(int dx) { this->x += dx; }` called as
  `p.move(1, 2)` (`this` is a `P *`; also works via `->`, nesting, and
  `this->other()` chaining). `union U { int i; half h; };` overlays all
  members at offset 0 (size = max); anonymous struct/union groups splice
  their members into the parent.

Templates (C++-style generics, monomorphized at parse time):
  `template <typename T> T max(T a, T b) {...}` used as `max<int>(3, 4)`;
  `template <typename T> struct Box { T val; };` used as
  `struct Box<int> b;`. Type args must be explicit (no deduction);
  templates must be defined before use; recursion works; unused broken
  templates are not diagnosed until instantiated. Template methods and
  non-type (value) parameters are not supported yet.

Defer: `defer f(x);` or `defer { ... }` runs the code when the enclosing
  scope ends (LIFO; also on return/break/continue, and per loop
  iteration). Deferred code is late-bound (sees variables as they are at
  run time) and shares the function's frame. No `return` inside deferred
  code. Needs the 256-entry runtime stack (emitted only when used).

`#include "other.b"` splices another file into the unit (relative to the
  including file; nested includes work; each file is included at most
  once). Only `#include` directives exist; errors report `file:line`.

Calling convention (M2): r10 = SP (0xFFFF down), r11 = FP, result in r0.
Args pushed right-to-left, caller cleans up; params at fp+(2+i), locals
below fp. All regs caller-saved. Builtins expand inline.

Memory map: user code+data lives in 0x0000..0x7FFF; 0x8000..0xFFFF is
reserved for libc's own use (stack grows down from 0xFFFF, future heap
and libc statics). Assemble user programs with:
    python3 bob16asm.py prog.asm --limit 0x8000

Pipeline: lex -> parse (recursive descent) -> AST -> assembler text
(mode-1 boot + mode-2), assembled by bob16asm.py.
"""
USER_TOP = 0x8000      # first reserved address (upper half is libc's)
import os
import re
import sys

# --------------------------------------------------------------------------
# lexer
# --------------------------------------------------------------------------
KEYWORDS = {'int', 'unsigned', 'char', 'void', 'half', 'struct', 'union', 'enum',
            'template', 'typename', 'defer',
            'if', 'else', 'while', 'do', 'for',
            'return', 'break', 'continue', 'sizeof', 'main'}

TOKEN_RE = re.compile(r'''
    (?P<ws>\s+)
  | (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<string>"(?:[^"\\\n]|\\.)*")
  | (?P<char>'(?:[^'\\\n]|\\.)*')
  | (?P<directive>\#[^\n]*)
  | (?P<half>\d+\.\d*(?:[eE][+-]?\d+)?h|\.\d+(?:[eE][+-]?\d+)?h|\d+[eE][+-]?\d+h|\d+h|(?i:inf(?:inity)?|nan)h)
  | (?P<num>0[xX][0-9a-fA-F]+|0[bB][01]+|\d+)
  | (?P<id>[A-Za-z_]\w*)
  | (?P<op><<=|>>=|==|!=|<=|>=|&&|\|\||<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|->|[+\-*/%<>=!&|^~?:;,.()\[\]{}])
''', re.S | re.X)


class Tok:
    def __init__(self, kind, text, pos, file=None, line=None, basedir=None):
        self.kind, self.text, self.pos = kind, text, pos
        self.file, self.line, self.basedir = file, line, basedir


def lex(text, fname='<string>', basedir='.'):
    toks, i, n = [], 0, len(text)
    while i < n:
        m = TOKEN_RE.match(text, i)
        if not m:
            line = text.count('\n', 0, i) + 1
            raise CompileError("bad character %r" % text[i], file=fname,
                               line=line)
        i = m.end()
        kind = m.lastgroup
        if kind in ('ws', 'comment'):
            continue
        t = m.group()
        if kind == 'id' and t in KEYWORDS:
            kind = t  # keyword token kind == keyword text
        line = text.count('\n', 0, m.start()) + 1
        toks.append(Tok(kind, t, m.start(), fname, line, basedir))
    toks.append(Tok('eof', '', n, fname, text.count('\n') + 1, basedir))
    return toks


class CompileError(Exception):
    def __init__(self, msg, file=None, line=None, pos=None):
        super().__init__(msg)
        self.file, self.line, self.pos = file, line, pos


# C escapes we accept (mapped onto what bob16asm.py understands)
CESc = {'n': '\\n', 't': '\\t', 'r': '\\r', '0': '\\0', '\\': '\\\\',
        "'": "\\'", '"': '\\"', 'a': '\\a', 'e': '\\e',
        'b': '\\x08', 'f': '\\x0c', 'v': '\\x0b'}


def c_str_body(raw):
    """raw includes quotes -> body re-escaped for bob16asm .string."""
    body, i, out = raw[1:-1], 0, []
    n = len(body)
    while i < n:
        c = body[i]
        if c == '\\':
            i += 1
            if i >= n:
                raise CompileError("dangling backslash in string")
            e = body[i]
            if e == 'x':
                out.append('\\x%s' % body[i + 1:i + 3])
                i += 2
            elif e in CESc:
                out.append(CESc[e])
            else:
                raise CompileError("unknown escape '\\%s'" % e)
        elif c == '"':
            out.append('\\"')
        else:
            out.append(c)
        i += 1
    return ''.join(out)


def c_char_val(raw):
    """'...' -> integer value."""
    body, i, out = raw[1:-1], 0, []
    n = len(body)
    while i < n:
        c = body[i]
        if c == '\\':
            i += 1
            e = body[i]
            if e == 'x':
                out.append(int(body[i + 1:i + 3], 16))
                i += 2
            elif e == 'n':
                out.append(10)
            elif e == 't':
                out.append(9)
            elif e == 'r':
                out.append(13)
            elif e == '0':
                out.append(0)
            elif e == 'a':
                out.append(7)
            elif e == 'b':
                out.append(8)
            elif e == 'f':
                out.append(12)
            elif e == 'v':
                out.append(11)
            elif e in ('\\', "'", '"'):
                out.append(ord(e))
            elif e == 'e':
                out.append(27)
            else:
                raise CompileError("unknown escape '\\%s'" % e)
        else:
            out.append(ord(c))
        i += 1
    if len(out) != 1:
        raise CompileError("character literal must hold exactly one char")
    return out[0]


def num_val(text):
    t = text.replace('#', '')
    if t[:2].lower() == '0x':
        return int(t, 16)
    if t[:2].lower() == '0b':
        return int(t, 2)
    if len(t) > 1 and t[0] == '0' and t[1:].isdigit():
        raise CompileError("octal literals not supported; use 0x")
    return int(t, 10)


# --------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------
class Num:
    def __init__(self, v):
        self.v = v


class HalfLit:
    def __init__(self, text):
        self.text = text  # e.g. '1.5h'; passed through to the assembler


class Str:
    def __init__(self, raw):
        self.raw = raw


class Var:
    def __init__(self, name):
        self.name = name


class Bin:
    def __init__(self, op, l, r):
        self.op, self.l, self.r = op, l, r


class Un:
    def __init__(self, op, x):
        self.op, self.x = op, x


class Cond:
    def __init__(self, c, t, f):
        self.c, self.t, self.f = c, t, f


class Assign:
    def __init__(self, op, target, val):
        self.op, self.target, self.val = op, target, val


class Sub:
    def __init__(self, base, idx):
        self.base, self.idx = base, idx


class Dot:
    def __init__(self, base, member, arrow=False):
        self.base, self.member, self.arrow = base, member, arrow  # base: Var/Sub/Deref/Dot


class Addr:
    def __init__(self, x):
        self.x = x


class Deref:
    def __init__(self, x):
        self.x = x


class Sizeof:
    def __init__(self, name):
        self.name = name  # type name; everything is 1 word


class SizeofType:
    def __init__(self, ty):
        self.ty = ty  # possibly symbolic (template param); resolved at stamp


class Builtin:
    def __init__(self, name, args):
        self.name, self.args = name, args

class FuncDef:
    def __init__(self, ret, name, params, body, src=None):
        self.ret, self.name, self.params, self.body = ret, name, params, body
        self.src = src


class Call:
    def __init__(self, name, args):
        self.name, self.args = name, args


class TCall:
    def __init__(self, name, typeargs, args):
        self.name, self.typeargs, self.args = name, typeargs, args


class MCall:
    def __init__(self, base, method, args, arrow):
        self.base, self.method, self.args, self.arrow = base, method, args, arrow


class Decl:
    def __init__(self, typename, name, arr, init, ptr=0, src=None):
        self.typename, self.name, self.arr, self.init, self.ptr = typename, name, arr, init, ptr
        self.src = src


class Block:
    def __init__(self, stmts):
        self.stmts = stmts


class If:
    def __init__(self, c, t, f):
        self.c, self.t, self.f = c, t, f


class While:
    def __init__(self, c, body):
        self.c, self.body = c, body


class DoWhile:
    def __init__(self, body, c):
        self.body, self.c = body, c


class For:
    def __init__(self, init, cond, incr, body):
        self.init, self.cond, self.incr, self.body = init, cond, incr, body


class Return:
    def __init__(self, val):
        self.val = val


class Break:
    pass


class Continue:
    pass


class ExprStmt:
    def __init__(self, e):
        self.e = e


class Defer:
    def __init__(self, stmt):
        self.stmt = stmt  # an ExprStmt or a Block, run at scope end (LIFO)


# --------------------------------------------------------------------------
# parser (recursive descent, C precedence)
# --------------------------------------------------------------------------
TYPES = {'int', 'unsigned', 'char', 'half'}

ASSIGN_OPS = {'=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=',
              '<<=', '>>='}


class Parser:
    def __init__(self, toks, basedir='.'):
        self.toks, self.pos = toks, 0
        self.basedir = basedir
        self.seen = set()   # realpaths already included (pragma-once)
        self.structs = {}   # name -> {'members': {m: (ty, ptr, arr, off, sz)}, 'size': n}
        self.unions = {}    # same shape as structs
        self.enums = {}     # enumerator -> int value
        self.enumtypes = set()
        self.tmpl_funcs = {}   # name -> (tparams, ret, params, body)
        self.tmpl_structs = {} # name -> (kind, tparams, raw_members)
        self.inst_funcs = {}   # (name, argtuple) -> instance name
        self.inst_structs = {} # (kind, name, argtuple) -> concrete name
        self.inst_active = set()
        self.cur_tparams = []  # type params in scope while parsing a template
        self.cur_funcs = None  # program funcs list (stamped instances append)

    def peek(self):
        return self.toks[self.pos]

    def next(self):
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def expect(self, kind):
        t = self.next()
        if t.kind != kind and t.text != kind:
            raise self.fail("expected '%s', got '%s'" % (kind, t.text))
        return t

    def at(self, *kinds):
        t = self.peek()
        return t.kind in kinds or t.text in kinds

    def fail(self, msg):
        t = self.peek()
        raise CompileError(msg, file=t.file, line=t.line)

    def do_include(self):
        """splice an #include file's tokens into the stream (pragma-once)."""
        t = self.next()  # the directive token
        m = re.match(r'^#\s*include\s*"([^"]+)"\s*$', t.text)
        if not m:
            raise CompileError("only #include \"file\" is supported",
                               file=t.file, line=t.line)
        inc = os.path.join(t.basedir or '.', m.group(1))
        rp = os.path.realpath(inc)
        if rp in self.seen:
            return
        try:
            with open(inc) as f:
                text = f.read()
        except OSError:
            raise CompileError("cannot open '%s'" % m.group(1),
                               file=t.file, line=t.line)
        self.seen.add(rp)
        sub = lex(text, inc, os.path.dirname(os.path.abspath(inc)))
        self.toks[self.pos:self.pos] = [x for x in sub if x.kind != 'eof']

    # ---- top level ----
    def parse_program(self):
        globals_, funcs, main = [], [], None
        self.cur_funcs = funcs
        while not self.at('eof'):
            if self.at('directive'):
                self.do_include()
                continue
            if self.at('template'):
                self.parse_template_def(funcs)
                continue
            if self.at(*TYPES, 'void'):
                src = self.peek().file
                ret = self.next().text
                fd = self.parse_top_decl(ret, src, globals_, funcs)
                if fd is not None and fd.name == 'main':
                    main = fd
            elif self.at('struct', 'union'):
                kind = self.next().text
                name = self.expect('id').text
                if self.at('{'):
                    self.parse_agg_body(name, kind)
                    self.expect(';')
                else:
                    table = self.structs if kind == 'struct' else self.unions
                    if name not in table:
                        raise self.fail("undefined %s '%s'" % (kind, name))
                    src = self.peek().file
                    fd = self.parse_top_decl(kind + ' ' + name, src, globals_, funcs)
                    if fd is not None and fd.name == 'main':
                        main = fd
            elif self.at('enum'):
                self.next()
                if self.at('{'):
                    self.parse_enum_body(None)
                    self.expect(';')
                else:
                    name = self.expect('id').text
                    if self.at('{'):
                        self.parse_enum_body(name)
                        self.expect(';')
                    else:
                        if name not in self.enumtypes:
                            raise self.fail("undefined enum '%s'" % name)
                        src = self.peek().file
                        fd = self.parse_top_decl('enum ' + name, src, globals_, funcs)
                        if fd is not None and fd.name == 'main':
                            main = fd
            else:
                raise self.fail("expected declaration or function, got '%s'"
                                   % self.peek().text)
        # lower template uses: instantiate concrete TCalls everywhere. New
        # instances appended to funcs are walked too (harmless re-entry).
        for fd in funcs:
            fd.body = self.transform(fd.body, None)
        for d in globals_:
            if d.init is not None and not isinstance(d.init, Str):
                d.init = self.transform(d.init, None)
        if main is None:
            raise self.fail("no `int main()` found")
        return globals_, funcs, main

    def parse_top_decl(self, ret, src, globals_, funcs):
        """shared tail: after the return/variable type, parse `name ...`."""
        ptr = 0
        while self.at('*'):
            self.next()
            ptr += 1
        t = self.next()
        if t.kind != 'id' and t.text != 'main':
            raise self.fail("expected a name, got '%s'" % t.text)
        name = t.text
        if self.at('<') and name in self.tmpl_structs:
            raise self.fail("methods on template instances are not supported")
        if name in self.tmpl_funcs or name in self.tmpl_structs:
            raise self.fail("'%s' is already defined" % name)
        if self.at('.'):
            return self.parse_method_def(ret, ptr, name, src, funcs)
        if self.at('('):
            if ret == 'void' and ptr:
                raise self.fail("bad return type")
            if ptr == 0 and (ret.startswith('struct ') or ret.startswith('union ')):
                raise self.fail("functions cannot return aggregates (return a pointer)")
            main = next((f for f in funcs if f.name == 'main'), None)
            if name == 'main' and (ret != 'int' or ptr or main is not None):
                raise self.fail("need exactly one `int main()`")
            params = self.parse_params(name == 'main')
            body = self.parse_block()
            fd = FuncDef(ret, name, params, body, src)
            funcs.append(fd)
            return fd
        else:
            if ret == 'void':
                raise self.fail("void variables not supported")
            arr = self.parse_arr_suffix()
            init = None
            if self.at('='):
                self.next()
                init = self.parse_init()
            self.expect(';')
            globals_.append(Decl(ret, name, arr, init, ptr, src))
            return None

    def parse_method_def(self, ret, ptr, aggname, src, funcs):
        """after `Ret Agg .`: parse `method(params) { body }` with implicit
        `this` (a pointer to the aggregate) as the first parameter."""
        kind = 'struct' if aggname in self.structs else \
               'union' if aggname in self.unions else None
        if kind is None:
            raise self.fail("undefined struct or union '%s'" % aggname)
        if ret == 'void' and ptr:
            raise self.fail("bad return type")
        if ptr == 0 and (ret.startswith('struct ') or ret.startswith('union ')):
            raise self.fail("methods cannot return aggregates (return a pointer)")
        self.next()  # '.'
        t = self.next()
        if t.kind != 'id' and t.text != 'main':
            raise self.fail("expected a method name, got '%s'" % t.text)
        key = aggname + '.' + t.text
        if any(f.name == key for f in funcs):
            raise self.fail("method '%s' already defined" % key)
        params = self.parse_params(False)
        params = [Decl(kind + ' ' + aggname, 'this', None, None, 1)] + params
        body = self.parse_block()
        fd = FuncDef(ret, key, params, body, src)
        funcs.append(fd)
        return fd

    def parse_template_def(self, funcs):
        """after `template`: `<typename T, ...>` then a function or a
        struct/union definition (stored uninstantiated)."""
        src = self.peek().file
        self.next()  # 'template'
        self.expect('<')
        tparams = []
        while True:
            self.expect('typename')
            tparams.append(self.expect('id').text)
            if not self.at(','):
                break
            self.next()
        self.expect('>')
        if len(set(tparams)) != len(tparams):
            raise self.fail("duplicate template parameter")
        save, self.cur_tparams = self.cur_tparams, tparams
        try:
            if self.at('struct', 'union'):
                kind = self.next().text
                name = self.expect('id').text
                self.check_tmpl_name(name)
                if not self.at('{'):
                    raise self.fail("template %ss need a body" % kind)
                # placeholder first: members may self-reference (pointers)
                self.tmpl_structs[name] = (kind, tparams, None)
                self.expect('{')
                raw = self.parse_raw_members(name)
                self.expect(';')
                if not raw:
                    raise self.fail("%s '%s' needs at least one member" % (kind, name))
                self.tmpl_structs[name] = (kind, tparams, raw)
                return
            ret = self.parse_typespec(allow_void=True)
            ptr = self.parse_stars()
            ret_ptr = ptr
            t = self.next()
            if t.kind != 'id' and t.text != 'main':
                raise self.fail("expected a name, got '%s'" % t.text)
            name = t.text
            if self.at('.'):
                raise self.fail("template methods are not supported")
            self.check_tmpl_name(name)
            if not self.at('('):
                raise self.fail("templates must be functions or aggregates")
            if ret == 'void' and ret_ptr:
                raise self.fail("bad return type")
            if ret_ptr == 0 and (ret.startswith('struct ') or ret.startswith('union ')):
                raise self.fail("functions cannot return aggregates (return a pointer)")
            # placeholder first: bodies may recurse into this template
            self.tmpl_funcs[name] = (tparams, None, 0, None, None, src)
            params = self.parse_params(False)
            body = self.parse_block()
            self.tmpl_funcs[name] = (tparams, ret, ret_ptr, params, body, src)
        finally:
            self.cur_tparams = save

    def check_tmpl_name(self, name):
        for f in self.cur_funcs:
            if f.name == name:
                raise self.fail("'%s' is already defined" % name)
        if name in self.tmpl_funcs or name in self.tmpl_structs or \
                name in self.enums or name in self.structs or \
                name in self.unions or name in self.enumtypes:
            raise self.fail("'%s' is already defined" % name)

    def parse_type_args(self):
        """parse `<T1, T2, ...>` (the `<` is consumed here; handles `>>`)."""
        self.expect('<')
        args = []
        if not self.at('>'):
            while True:
                args.append(self.parse_typespec())
                if not self.at(','):
                    break
                self.next()
        self.expect_gt()
        return args

    def expect_gt(self):
        """consume one `>`, splitting a `>>` token when nested."""
        t = self.peek()
        if t.text == '>>':
            t1 = Tok('op', '>', t.pos, t.file, t.line, t.basedir)
            t2 = Tok('op', '>', t.pos + 1, t.file, t.line, t.basedir)
            self.toks[self.pos:self.pos + 1] = [t1, t2]
        self.expect('>')

    @staticmethod
    def is_symbolic(ty, scope):
        return any(w in scope for w in re.findall(r'[A-Za-z_]\w*', ty))

    @staticmethod
    def split_tid(ty):
        """split 'struct Box<T, int>' -> (('struct', 'Box'), ['T', 'int'])."""
        i = ty.find('<')
        if i < 0 or not ty.endswith('>'):
            return (None, None)
        pre, inner = ty[:i], ty[i + 1:-1]
        parts = pre.split(' ', 1)
        if len(parts) != 2:
            return (None, None)
        args, depth, cur = [], 0, ''
        for c in inner:
            if c == '<':
                depth += 1
                cur += c
            elif c == '>':
                depth -= 1
                cur += c
            elif c == ',' and depth == 0:
                args.append(cur)
                cur = ''
            else:
                cur += c
        args.append(cur)
        return ((parts[0], parts[1]), args)

    def subst_type(self, ty, env):
        """resolve a possibly-symbolic typename with a tparam env. Concrete
        template-ids are instantiated on the spot."""
        if env and ty in env:
            return env[ty]
        base, args = self.split_tid(ty)
        if base is None:
            return ty
        kind, tname = base
        newargs = [self.subst_type(a, env) for a in args]
        scope = set(env) if env else set()
        if all(not self.is_symbolic(a, scope) for a in newargs):
            self.ensure_struct_instance(kind, tname, newargs)
            return '%s %s<%s>' % (kind, tname, ','.join(newargs))
        return '%s %s<%s>' % (kind, tname, ','.join(newargs))

    def ensure_struct_instance(self, kind, tname, argtypes):
        """lay out a template aggregate for concrete args; returns its name."""
        key = (kind, tname, tuple(argtypes))
        if key in self.inst_structs:
            return self.inst_structs[key]
        t = self.tmpl_structs.get(tname)
        if t is None or t[0] != kind:
            raise self.fail("'%s' is not a %s template" % (tname, kind))
        if t[2] is None:
            raise self.fail("cannot instantiate '%s' inside its own body" % tname)
        if key in self.inst_active:
            raise self.fail("recursive aggregate instantiation")
        display = tname + '<' + ','.join(argtypes) + '>'
        self.inst_active.add(key)
        try:
            tparams, raw = t[1], t[2]
            env = dict(zip(tparams, argtypes))
            craw = [(self.subst_type(ty, env), ptr, arr, mname)
                    for (ty, ptr, arr, mname) in raw]
            members, size = self.layout_raw(kind, craw, display)
        finally:
            self.inst_active.discard(key)
        table = self.structs if kind == 'struct' else self.unions
        table[display] = {'members': members, 'size': size}
        self.inst_structs[key] = display
        return display

    def ensure_func_instance(self, name, argtypes):
        """stamp out a template function for concrete args; returns its name."""
        key = (name, tuple(argtypes))
        if key in self.inst_funcs:
            return self.inst_funcs[key]
        t = self.tmpl_funcs.get(name)
        if t is None:
            raise self.fail("'%s' is not a function template" % name)
        tparams, ret, ret_ptr, params, body, src = t
        if ret is None:
            raise self.fail("cannot instantiate '%s' inside its own body" % name)
        if len(argtypes) != len(tparams):
            raise self.fail("'%s' takes %d type argument(s), got %d"
                            % (name, len(tparams), len(argtypes)))
        display = name + '<' + ','.join(argtypes) + '>'
        if key in self.inst_active:
            return display  # recursive call; the FuncDef is being stamped
        self.inst_active.add(key)
        try:
            env = dict(zip(tparams, argtypes))
            newret = self.subst_type(ret, env)
            if ret_ptr == 0 and (newret.startswith('struct ') or
                                 newret.startswith('union ')):
                raise self.fail("functions cannot return aggregates (return a pointer)")
            newparams = [Decl(self.subst_type(p.typename, env), p.name,
                              p.arr, None, p.ptr, p.src) for p in params]
            for p in newparams:
                if (p.typename.startswith('struct ') or
                        p.typename.startswith('union ')) and \
                        p.ptr == 0 and p.arr != 0:
                    raise self.fail("%s params must be pointers"
                                    % p.typename.split(' ', 1)[0])
            newbody = self.transform(body, env)
            fd = FuncDef(newret, display, newparams, newbody, src)
            self.cur_funcs.append(fd)
        finally:
            self.inst_active.discard(key)
        self.inst_funcs[key] = display
        return display

    def transform(self, node, env):
        """clone AST resolving templates. env=None: instantiate concrete
        uses, keep symbolic ones. env=dict: stamp (all must resolve)."""
        if isinstance(node, Decl):
            init = self.transform(node.init, env) if node.init is not None else None
            return Decl(self.subst_type(node.typename, env), node.name,
                        node.arr, init, node.ptr, node.src)
        if isinstance(node, Block):
            return Block([self.transform(s, env) for s in node.stmts])
        if isinstance(node, If):
            return If(self.transform(node.c, env), self.transform(node.t, env),
                       self.transform(node.f, env) if node.f is not None else None)
        if isinstance(node, While):
            return While(self.transform(node.c, env), self.transform(node.body, env))
        if isinstance(node, DoWhile):
            return DoWhile(self.transform(node.body, env), self.transform(node.c, env))
        if isinstance(node, For):
            init = node.init
            if isinstance(init, Decl):
                init = self.transform(init, env)
            elif init is not None:
                init = self.transform(init, env)
            return For(init,
                       self.transform(node.cond, env) if node.cond is not None else None,
                       self.transform(node.incr, env) if node.incr is not None else None,
                       self.transform(node.body, env))
        if isinstance(node, Return):
            return Return(self.transform(node.val, env) if node.val is not None else None)
        if isinstance(node, (Break, Continue)):
            return node
        if isinstance(node, ExprStmt):
            return ExprStmt(self.transform(node.e, env))
        if isinstance(node, Defer):
            return Defer(self.transform(node.stmt, env))
        if isinstance(node, (Num, HalfLit, Str, Var, Sizeof)):
            return node
        if isinstance(node, SizeofType):
            if env is None:
                return node
            return Num(self.concrete_sizeof(self.subst_type(node.ty, env)))
        if isinstance(node, Bin):
            return Bin(node.op, self.transform(node.l, env), self.transform(node.r, env))
        if isinstance(node, Un):
            return Un(node.op, self.transform(node.x, env))
        if isinstance(node, Cond):
            return Cond(self.transform(node.c, env), self.transform(node.t, env),
                        self.transform(node.f, env))
        if isinstance(node, Assign):
            return Assign(node.op, self.transform(node.target, env),
                          self.transform(node.val, env))
        if isinstance(node, Sub):
            return Sub(self.transform(node.base, env), self.transform(node.idx, env))
        if isinstance(node, Dot):
            return Dot(self.transform(node.base, env), node.member, node.arrow)
        if isinstance(node, Addr):
            return Addr(self.transform(node.x, env))
        if isinstance(node, Deref):
            return Deref(self.transform(node.x, env))
        if isinstance(node, Builtin):
            return Builtin(node.name, [self.transform(a, env) for a in node.args])
        if isinstance(node, Call):
            return Call(node.name, [self.transform(a, env) for a in node.args])
        if isinstance(node, MCall):
            return MCall(self.transform(node.base, env), node.method,
                         [self.transform(a, env) for a in node.args], node.arrow)
        if isinstance(node, TCall):
            scope = set(env) if env else set()
            newargs = [self.subst_type(t, env) for t in node.typeargs]
            targs = [self.transform(a, env) for a in node.args]
            if all(not self.is_symbolic(t, scope) for t in newargs):
                return Call(self.ensure_func_instance(node.name, newargs), targs)
            if env is not None:
                raise self.fail("cannot resolve template arguments")
            return TCall(node.name, newargs, targs)
        raise self.fail("cannot transform that node")

    def concrete_sizeof(self, ty):
        base, args = self.split_tid(ty)
        if base is None:
            return 1  # scalar or (already rejected) param
        kind, sname = base
        table = self.structs if kind == 'struct' else self.unions
        if sname not in table:
            raise self.fail("undefined %s '%s'" % (kind, sname))
        return table[sname]['size']

    def parse_agg_body(self, name, kind):
        """define a named struct/union; kind is 'struct' or 'union'."""
        table = self.structs if kind == 'struct' else self.unions
        if name in table:
            raise self.fail("%s '%s' already defined" % (kind, name))
        if name in self.tmpl_funcs or name in self.tmpl_structs:
            raise self.fail("'%s' is already defined" % name)
        self.expect('{')
        members, size = self.parse_member_list(kind, name)
        if not members:
            raise self.fail("%s '%s' needs at least one member" % (kind, name))
        table[name] = {'members': members, 'size': size}

    def parse_raw_members(self, selfname):
        """member decls for template bodies: [(ty, ptr, arr, name)], no layout
        (sizes may depend on type params). Anonymous groups are rejected."""
        raw = []
        seen = set()
        while not self.at('}'):
            if self.at('struct', 'union') and self.toks[self.pos + 1].text == '{':
                raise self.fail("anonymous groups need concrete types")
            ty = self.parse_typespec(selfstruct=selfname)
            ptr = self.parse_stars()
            mname = self.expect('id').text
            if mname in seen:
                raise self.fail("duplicate member '%s'" % mname)
            seen.add(mname)
            arr = self.parse_arr_suffix()
            if self.at('='):
                raise self.fail("no initializers inside aggregates")
            self.expect(';')
            raw.append((ty, ptr, arr, mname))
        self.expect('}')
        return raw

    def layout_raw(self, kind, raw, selfname):
        """lay out raw members -> (members dict, total size)."""
        members, off, mx = {}, 0, 0
        for (ty, ptr, arr, mname) in raw:
            sz = self.member_size(ty, ptr, arr, selfname)
            if kind == 'struct':
                members[mname] = (ty, ptr, arr, off, sz)
                off += sz
            else:
                members[mname] = (ty, ptr, arr, 0, sz)
                mx = max(mx, sz)
        return (members, off if kind == 'struct' else mx)

    def parse_member_list(self, kind, selfname):
        """members until '}' (consumed); returns (members, total size).
        Allows anonymous struct/union groups spliced at the current offset."""
        members, off, mx = {}, 0, 0
        while not self.at('}'):
            if self.at('struct', 'union') and self.toks[self.pos + 1].text == '{':
                gkind = self.next().text
                self.expect('{')
                sub, subsize = self.parse_member_list(gkind, selfname)
                self.expect(';')
                for sn, (ty, ptr, arr, ro, sz) in sub.items():
                    if sn in members:
                        raise self.fail("duplicate member '%s'" % sn)
                    if kind == 'struct':
                        members[sn] = (ty, ptr, arr, off + ro, sz)
                    else:
                        members[sn] = (ty, ptr, arr, ro, sz)
                if kind == 'struct':
                    off += subsize
                else:
                    mx = max(mx, subsize)
                continue
            ty = self.parse_typespec(selfstruct=selfname)
            ptr = self.parse_stars()
            mname = self.expect('id').text
            if mname in members:
                raise self.fail("duplicate member '%s'" % mname)
            arr = self.parse_arr_suffix()
            if self.at('='):
                raise self.fail("no initializers inside %ss" % kind)
            self.expect(';')
            sz = self.member_size(ty, ptr, arr, selfname)
            if kind == 'struct':
                members[mname] = (ty, ptr, arr, off, sz)
                off += sz
            else:
                members[mname] = (ty, ptr, arr, 0, sz)
                mx = max(mx, sz)
        self.expect('}')
        return (members, off if kind == 'struct' else mx)

    def member_size(self, ty, ptr, arr, selfstruct=None):
        if ptr > 0:
            return 1
        if ty.startswith('struct ') or ty.startswith('union '):
            kind, sname = ty.split(' ', 1)
            if sname == selfstruct:
                raise self.fail("cannot nest %s '%s' by value" % (kind, sname))
            table = self.structs if kind == 'struct' else self.unions
            elem = table[sname]['size']
        else:
            elem = 1
        return (arr or 1) * elem

    def parse_enum_body(self, name):
        if name is not None:
            if name in self.enumtypes:
                raise self.fail("enum '%s' already defined" % name)
            if name in self.tmpl_funcs or name in self.tmpl_structs:
                raise self.fail("'%s' is already defined" % name)
            self.enumtypes.add(name)
        self.expect('{')
        val = 0
        seen_any = False
        while not self.at('}'):
            seen_any = True
            ename = self.expect('id').text
            if ename in self.enums:
                raise self.fail("enumerator '%s' already defined" % ename)
            if ename in self.tmpl_funcs or ename in self.tmpl_structs:
                raise self.fail("'%s' is already defined" % ename)
            if self.at('='):
                self.next()
                val = self.fold_int(self.parse_assign())
            self.enums[ename] = val
            val += 1
            if not self.at(','):
                break
            self.next()
            if self.at('}'):
                break  # trailing comma
        if not seen_any:
            raise self.fail("enum needs at least one enumerator")
        self.expect('}')

    def fold_int(self, node):
        """parse-time integer constant folding (16-bit words)."""
        if isinstance(node, Num):
            return node.v & 0xFFFF
        if isinstance(node, Sizeof):
            return 1
        if isinstance(node, Var):
            if node.name in self.enums:
                return self.enums[node.name]
            raise self.fail("'%s' is not a constant" % node.name)
        if isinstance(node, Un) and node.op == '-':
            return (-self.fold_int(node.x)) & 0xFFFF
        if isinstance(node, Un) and node.op == '~':
            return (~self.fold_int(node.x)) & 0xFFFF
        if isinstance(node, Un) and node.op == '+':
            return self.fold_int(node.x)
        if isinstance(node, Un) and node.op == '!':
            return 0 if self.fold_int(node.x) else 1
        if isinstance(node, Bin):
            a, b = self.fold_int(node.l), self.fold_int(node.r)
            sa = lambda v: v - 65536 if v >= 32768 else v
            op = node.op
            if op == '+':
                return (a + b) & 0xFFFF
            if op == '-':
                return (a - b) & 0xFFFF
            if op == '*':
                return (a * b) & 0xFFFF
            if op == '/':
                return int(sa(a) / sa(b)) & 0xFFFF if b else self.fail("division by zero")
            if op == '%':
                if not b:
                    raise self.fail("division by zero")
                q = abs(sa(a)) // abs(sa(b))
                r = abs(sa(a)) - abs(sa(b)) * q
                return ((r if sa(a) >= 0 else -r) & 0xFFFF)
            if op == '<<':
                return (a << b) & 0xFFFF
            if op == '>>':
                return (sa(a) >> b) & 0xFFFF
            if op == '&':
                return a & b
            if op == '|':
                return a | b
            if op == '^':
                return a ^ b
            if op in ('==', '!=', '<', '>', '<=', '>='):
                av, bv = sa(a), sa(b)
                r = {'==': av == bv, '!=': av != bv, '<': av < bv,
                     '>': av > bv, '<=': av <= bv, '>=': av >= bv}[op]
                return 1 if r else 0
            if op == '&&':
                return 1 if (a and b) else 0
            if op == '||':
                return 1 if (a or b) else 0
        raise self.fail("not a constant expression")

    def parse_typespec(self, allow_void=False, selfstruct=None):
        """a type name: scalar, `struct S` / `union U` (must be defined) or
        `enum E`."""
        if self.at('struct', 'union'):
            kind = self.next().text
            name = self.expect('id').text
            if self.at('<'):
                t = self.tmpl_structs.get(name)
                if t is None or t[0] != kind:
                    raise self.fail("'%s' is not a %s template" % (name, kind))
                args = self.parse_type_args()
                if len(args) != len(t[1]):
                    raise self.fail("'%s' takes %d type argument(s), got %d"
                                    % (name, len(t[1]), len(args)))
                key = name + '<' + ','.join(args) + '>'
                scope = set(self.cur_tparams)
                if all(not self.is_symbolic(a, scope) for a in args):
                    self.ensure_struct_instance(kind, name, args)
                return kind + ' ' + key
            table = self.structs if kind == 'struct' else self.unions
            if name not in table and name != selfstruct:
                raise self.fail("undefined %s '%s'" % (kind, name))
            return kind + ' ' + name
        if self.at('enum'):
            self.next()
            name = self.expect('id').text
            if name not in self.enumtypes:
                raise self.fail("undefined enum '%s'" % name)
            return 'enum ' + name
        t = self.next()
        if t.text in self.cur_tparams:
            return t.text
        if t.text == 'void' and allow_void:
            return 'void'
        if t.text in TYPES:
            return t.text
        raise self.fail("expected a type, got '%s'" % t.text)

    def parse_params(self, is_main):
        self.expect('(')
        params = []
        if self.at('void'):
            self.next()
            if not self.at(')'):
                raise self.fail("`void` means no parameters")
        elif not self.at(')'):
            while True:
                t = self.peek()
                if not self.at(*TYPES, 'struct', 'enum', 'union') and \
                        t.text not in self.cur_tparams:
                    raise self.fail("parameter needs a type")
                ty = self.parse_typespec()
                ptr = self.parse_stars()
                pname = self.expect('id').text
                arr = None
                if self.at('['):
                    self.next()
                    if not self.at(']'):
                        self.expect('num')
                    self.expect(']')
                    arr = 0  # arrays decay to pointers (1 slot)
                if (ty.startswith('struct ') or ty.startswith('union ')) \
                        and ptr == 0 and arr != 0:
                    raise self.fail("%s params must be pointers (use %s *%s)"
                                    % (ty.split(' ', 1)[0], ty, pname))
                params.append(Decl(ty, pname, arr, None, ptr or 1 if arr == 0 else ptr))
                if not self.at(','):
                    break
                self.next()
        self.expect(')')
        if is_main and params:
            raise self.fail("main takes no arguments in M2")
        return params

    def parse_arr_suffix(self):
        if self.at('['):
            self.next()
            n = self.fold_int(self.parse_assign())
            self.expect(']')
            if n < 1 or n > 0xFFFF:
                raise self.fail("array size must be 1..65535")
            return n
        return None

    def parse_init(self):
        if self.at('string'):
            return Str(self.next().text)
        return self.parse_assign()

    def parse_block(self):
        self.expect('{')
        stmts = []
        while not self.at('}'):
            stmts.append(self.parse_stmt())
        self.expect('}')
        return Block(stmts)

    def parse_stmt_or_decl(self):
        if self.at(*TYPES):
            return self.parse_decl(require_semi=True)
        return self.parse_stmt()

    def parse_stars(self):
        n = 0
        while self.at('*'):
            self.next()
            n += 1
        return n

    def parse_decl(self, require_semi):
        src = self.peek().file
        ty = self.parse_typespec()
        ptr = self.parse_stars()
        name = self.expect('id').text
        arr = self.parse_arr_suffix()
        init = None
        if self.at('='):
            self.next()
            init = self.parse_init()
        if require_semi:
            self.expect(';')
        return Decl(ty, name, arr, init, ptr, src)

    def parse_stmt(self):
        while self.at('directive'):
            self.do_include()
        if self.at(*TYPES):
            return self.parse_decl(require_semi=True)
        if self.at('struct', 'enum', 'union'):
            # local `struct P ...` / `enum E ...` declaration (not a definition)
            if self.toks[self.pos + 2].text == '{' or \
                    (self.toks[self.pos + 1].text == '{'):
                raise self.fail("struct/enum/union definitions go at top level")
            return self.parse_decl(require_semi=True)
        if self.at('{'):
            return self.parse_block()
        if self.at('if'):
            self.next()
            self.expect('(')
            c = self.parse_assign()
            self.expect(')')
            t = self.parse_body_block()
            f = None
            if self.at('else'):
                self.next()
                f = self.parse_body_block()
            return If(c, t, f)
        if self.at('while'):
            self.next()
            self.expect('(')
            c = self.parse_assign()
            self.expect(')')
            return While(c, self.parse_body_block())
        if self.at('do'):
            self.next()
            body = self.parse_body_block()
            self.expect('while')
            self.expect('(')
            c = self.parse_assign()
            self.expect(')')
            self.expect(';')
            return DoWhile(body, c)
        if self.at('for'):
            self.next()
            self.expect('(')
            init = None
            if not self.at(';'):
                if self.at(*TYPES, 'struct', 'enum', 'union'):
                    init = self.parse_decl(require_semi=False)
                else:
                    init = ExprStmt(self.parse_assign())
            self.expect(';')
            cond = None
            if not self.at(';'):
                cond = self.parse_assign()
            self.expect(';')
            incr = None
            if not self.at(')'):
                incr = self.parse_assign()
            self.expect(')')
            return For(init, cond, incr, self.parse_body_block())
        if self.at('defer'):
            self.next()
            if self.at('{'):
                return Defer(self.parse_block())
            if self.at('id') and self.toks[self.pos + 1].text == '(':
                if self.peek().text in ('puts', 'putc', 'print_int', 'gettime', 'settime'):
                    b = self.parse_builtin()
                    self.expect(';')
                    return Defer(ExprStmt(b))
            e = self.parse_assign()
            self.expect(';')
            return Defer(ExprStmt(e))
        if self.at('return'):
            self.next()
            v = None
            if not self.at(';'):
                v = self.parse_assign()
            self.expect(';')
            return Return(v)
        if self.at('break'):
            self.next()
            self.expect(';')
            return Break()
        if self.at('continue'):
            self.next()
            self.expect(';')
            return Continue()
        if self.at('id') and self.toks[self.pos + 1].text == '(':
            if self.peek().text in ('puts', 'putc', 'print_int', 'gettime', 'settime'):
                b = self.parse_builtin()
                self.expect(';')
                return ExprStmt(b)
        e = self.parse_assign()
        self.expect(';')
        return ExprStmt(e)

    def parse_body_block(self):
        """a branch/loop body: single statements wrap in a Block so every
        body is a scope (deferred code needs the anchor)."""
        s = self.parse_stmt()
        return s if isinstance(s, Block) else Block([s])

    def parse_builtin(self):
        name = self.next().text
        if name not in ('puts', 'putc', 'print_int', 'gettime', 'settime'):
            raise self.fail("unknown function '%s'" % name)
        self.expect('(')
        args = []
        if not self.at(')'):
            args.append(self.parse_assign())
        self.expect(')')
        return Builtin(name, args)

    # ---- expressions ----
    def parse_assign(self):
        e = self.parse_cond()
        if self.peek().text in ASSIGN_OPS:
            op = self.next().text
            if not isinstance(e, (Var, Sub, Deref, Dot)):
                raise self.fail("assignment target must be a variable, a[i], *p or s.m")
            v = self.parse_assign()
            return Assign(op, e, v)
        return e

    def parse_cond(self):
        e = self.parse_lor()
        if self.at('?'):
            self.next()
            t = self.parse_assign()
            self.expect(':')
            f = self.parse_cond()
            return Cond(e, t, f)
        return e

    def _bin(self, sub, ops):
        e = sub()
        while self.peek().text in ops:
            op = self.next().text
            r = sub()
            e = Bin(op, e, r)
        return e

    def parse_lor(self):
        return self._bin(self.parse_land, {'||'})

    def parse_land(self):
        return self._bin(self.parse_bor, {'&&'})

    def parse_bor(self):
        return self._bin(self.parse_bxor, {'|'})

    def parse_bxor(self):
        return self._bin(self.parse_band, {'^'})

    def parse_band(self):
        return self._bin(self.parse_eq, {'&'})

    def parse_eq(self):
        return self._bin(self.parse_rel, {'==', '!='})

    def parse_rel(self):
        return self._bin(self.parse_shift, {'<', '>', '<=', '>='})

    def parse_shift(self):
        return self._bin(self.parse_add, {'<<', '>>'})

    def parse_add(self):
        return self._bin(self.parse_mul, {'+', '-'})

    def parse_mul(self):
        return self._bin(self.parse_unary, {'*', '/', '%'})

    def parse_unary(self):
        if self.at('-', '+', '!', '~', '*', '&'):
            op = self.next().text
            x = self.parse_unary()
            if op == '+':
                return x
            if op == '*':
                return Deref(x)
            if op == '&':
                return Addr(x)
            return Un(op, x)
        return self.parse_postfix()

    def parse_call_args(self, node):
        """after `name`/`base.method`: parse `(args)` into a Call/MCall."""
        self.next()
        args = []
        if not self.at(')'):
            args.append(self.parse_assign())
            while self.at(','):
                self.next()
                args.append(self.parse_assign())
        self.expect(')')
        node.args = args
        return node

    def parse_postfix(self):
        e = self.parse_primary()
        while True:
            if self.at('['):
                self.next()
                idx = self.parse_assign()
                self.expect(']')
                e = Sub(e, idx)
            elif self.at('(') and isinstance(e, Var):
                self.next()
                args = []
                if not self.at(')'):
                    args.append(self.parse_assign())
                    while self.at(','):
                        self.next()
                        args.append(self.parse_assign())
                self.expect(')')
                e = Call(e.name, args)
            elif isinstance(e, Var) and e.name in self.tmpl_funcs and self.at('<'):
                # `f<int>(...)`: try template-id, else rewind (it's `<`)
                save = (self.pos, list(self.toks))
                try:
                    targs = self.parse_type_args()
                except CompileError:
                    targs = None
                if targs is None or not self.at('('):
                    self.pos, self.toks = save
                    return e
                t = self.tmpl_funcs[e.name]
                if len(targs) != len(t[0]):
                    raise self.fail("'%s' takes %d type argument(s), got %d"
                                    % (e.name, len(t[0]), len(targs)))
                e = self.parse_call_args(TCall(e.name, targs, []))
            elif self.at('.'):
                if isinstance(e, Num):
                    raise self.fail("float literals need an `h` suffix (e.g. 1.5h)")
                self.next()
                member = self.expect('id').text
                if self.at('('):
                    e = self.parse_call_args(MCall(e, member, [], False))
                else:
                    e = Dot(e, member)
            elif self.at('->'):
                self.next()
                member = self.expect('id').text
                if self.at('('):
                    e = self.parse_call_args(MCall(Deref(e), member, [], True))
                else:
                    e = Dot(Deref(e), member, True)
            else:
                return e

    def parse_primary(self):
        t = self.peek()
        if t.kind == 'num':
            self.next()
            return Num(num_val(t.text))
        if t.kind == 'half':
            self.next()
            return HalfLit(t.text)
        if t.kind == 'char':
            self.next()
            return Num(c_char_val(t.text))
        if t.kind == 'string':
            self.next()
            return Str(t.text)
        if t.kind == 'sizeof':
            self.next()
            self.expect('(')
            if self.at('struct', 'union'):
                kind = self.next().text
                name = self.expect('id').text
                if self.at('<'):
                    t = self.tmpl_structs.get(name)
                    if t is None or t[0] != kind:
                        raise self.fail("'%s' is not a %s template" % (name, kind))
                    args = self.parse_type_args()
                    if len(args) != len(t[1]):
                        raise self.fail("'%s' takes %d type argument(s), got %d"
                                        % (name, len(t[1]), len(args)))
                    key = name + '<' + ','.join(args) + '>'
                    scope = set(self.cur_tparams)
                    self.expect(')')
                    if all(not self.is_symbolic(a, scope) for a in args):
                        cname = self.ensure_struct_instance(kind, name, args)
                        table = self.structs if kind == 'struct' else self.unions
                        return Num(table[cname]['size'])
                    return SizeofType(kind + ' ' + key)
                table = self.structs if kind == 'struct' else self.unions
                if name not in table:
                    raise self.fail("undefined %s '%s'" % (kind, name))
                self.expect(')')
                return Num(table[name]['size'])
            if self.at('enum'):
                self.next()
                name = self.expect('id').text
                if name not in self.enumtypes:
                    raise self.fail("undefined enum '%s'" % name)
                self.expect(')')
                return Num(1)
            ty = self.next()
            if ty.text in self.cur_tparams:
                self.expect(')')
                return SizeofType(ty.text)
            if ty.text not in TYPES:
                raise self.fail("sizeof needs a type name")
            self.expect(')')
            return Sizeof(ty.text)
        if t.kind == 'id' or t.text in ('main',):
            self.next()
            return Var(t.text)
        if self.at('('):
            self.next()
            e = self.parse_assign()
            self.expect(')')
            return e
        raise self.fail("expected an expression, got '%s'" % t.text)


# --------------------------------------------------------------------------
# codegen (stack machine: every expression leaves its value in r0)
# --------------------------------------------------------------------------
class Gen:
    def __init__(self):
        self.out = []
        self.data = []
        self.nlabel = 0
        self.nstr = 0
        self.locals = {}      # name -> slot index (per function)
        self.nslots = 0
        self.params = {}      # name -> param index (per function)
        self.globals = {}     # name -> (label, size)
        self.arrays = set()   # local array names (decay to &slot0 as values)
        self.garrays = set()  # global array names
        self.funcs = {}       # name -> FuncDef
        self.loops = []       # (continue_label, break_label, bodymark|None)
        self.end = 'Lend'
        self.retend = 'Lret'
        self.cur_void = False
        self.cur_ret = 'int'
        self.cur_file = None    # for error messages in multi-file units
        self.cur_func = None
        self.local_sizes = {} # local name -> total words (per function)
        self.vartype = {}     # local/param name -> (typename, ptrdepth, is_array)
        self.gvartype = {}    # global name -> (typename, ptrdepth, is_array)
        self.structs = {}     # from parser: name -> {members, size}
        self.unions = {}      # same shape as structs
        self.enums = {}       # from parser: enumerator -> value
        self.func_labels = {} # asm label -> function key (collision guard)
        self.defer_seq = 0    # program-wide deferred-body counter
        self.deferred = []    # Defer nodes pending end-of-program emission
        self.need_defer = False
        self.in_defer = 0
        self.func_defer = False
        self.base_mark = None
        self.scratch_base = 0
        self.scratch_size = 0
        self.defer_sites = []
        # defer runtime globals (emitted only when used)
        self.DSTACK = 'G_defer_stack'
        self.DTOP = 'G_defer_top'
        self.DMAX = 256

    def fp_off(self, name):
        """frame operand offset from r11: params +(2+i); locals address the
        BOTTOM of their slot range so array/struct indexing (base+i, like
        globals) stays inside the frame: -(slot+size)."""
        if name in self.params:
            return 2 + self.params[name]
        if name in self.locals:
            return -(self.locals[name] + self.local_sizes.get(name, 1))
        raise self.fail("undefined variable '%s'" % name)

    def is_frame(self, name):
        return name in self.params or name in self.locals

    def frame_op(self, name):
        return 'r11%+d' % self.fp_off(name)

    def gen_load_var(self, name):
        if name in self.enums and name not in self.locals \
                and name not in self.params and name not in self.globals:
            self.emit('move r0, %d' % self.enums[name])
            return
        if self.is_frame(name):
            self.emit('load r0, %s' % self.frame_op(name))
        elif name in self.globals:
            self.emit('load r0, %s' % self.globals[name][0])
        else:
            raise self.fail("undefined variable '%s'" % name)

    def gen_value_var(self, name):
        """rvalue of a variable: arrays decay to their address."""
        if name in self.params:
            self.gen_load_var(name)
        elif name in self.locals:
            if name in self.arrays:
                self.gen_addr_var(name)
            else:
                self.gen_load_var(name)
        elif name in self.globals:
            if name in self.garrays:
                self.emit('move r0, %s' % self.globals[name][0])
            else:
                self.gen_load_var(name)
        else:
            raise self.fail("undefined variable '%s'" % name)

    def gen_addr_var(self, name):
        if self.is_frame(name):
            self.emit('move r0, %s' % self.frame_op(name))
        elif name in self.globals:
            self.emit('move r0, %s' % self.globals[name][0])
        else:
            raise self.fail("undefined variable '%s'" % name)

    def gen_store_var(self, name):
        """r0 holds the value."""
        if name in self.enums:
            raise self.fail("cannot assign to enumerator '%s'" % name)
        if self.is_frame(name):
            op = self.frame_op(name)
        elif name in self.globals:
            op = None
        else:
            raise self.fail("undefined variable '%s'" % name)
        self.emit('push r10, r0')
        if op is None:
            self.emit('move r1, %s' % self.globals[name][0])
        else:
            self.emit('move r1, %s' % op)
        self.emit('pop r0, r10')
        self.emit('stor r1, r0')

    # ---- M3 static types ('int' means any integer type; addresses are int) ----
    def _vt(self, name):
        if name in self.vartype:
            return self.vartype[name]
        if name in self.gvartype:
            return self.gvartype[name]
        raise self.fail("undefined variable '%s'" % name)

    def ptr_target(self, node):
        """pointed-to typename if node is (pointer arithmetic on) a typed
        pointer variable, else None. Pointers are plain ints at runtime.
        Aggregate-typed calls always return pointers (by value is banned)."""
        if isinstance(node, Var):
            try:
                ty, ptr, arr = self._vt(node.name)
            except CompileError:
                return None
            return ty if ptr > 0 else None
        if isinstance(node, Bin) and node.op in ('+', '-'):
            return self.ptr_target(node.l) or self.ptr_target(node.r)
        if isinstance(node, Call):
            fd = self.funcs.get(node.name)
            if fd is not None and self.agg(fd.ret)[0] is not None:
                return fd.ret
            return None
        if isinstance(node, MCall):
            try:
                fd = self.mcall_target(node)[2]
            except CompileError:
                return None
            if self.agg(fd.ret)[0] is not None:
                return fd.ret
            return None
        return None

    @staticmethod
    def norm(ty):
        return 'half' if ty == 'half' else 'int'

    def rtype(self, node):
        """type of an expression's value in r0: 'int' or 'half'."""
        if isinstance(node, HalfLit):
            return 'half'
        if isinstance(node, (Num, Str, Sizeof, Addr)):
            return 'int'
        if isinstance(node, Var):
            if node.name in self.enums and node.name not in self.locals \
                    and node.name not in self.params \
                    and node.name not in self.globals:
                return 'int'
            a = self.var_agg(node.name)
            if a is not None:
                raise self.fail(self.agg_val_err(a))
            ty, ptr, arr = self._vt(node.name)
            return 'int' if (ptr > 0 or arr) else self.norm(ty)
        if isinstance(node, Sub):
            a = self.sub_elem_struct(node.base)
            if a is not None:
                raise self.fail(self.agg_val_err(a))
            base = node.base
            if isinstance(base, Var):
                return self.norm(self._vt(base.name)[0])
            if isinstance(base, Dot):
                dm = self.dot_member(base)
                if dm is None:
                    raise self.fail(self.dot_err(base))
                return self.norm(dm[2][0])
            raise self.fail("array base must be a variable")
        if isinstance(node, Dot):
            dm = self.dot_member(node)
            if dm is None:
                raise self.fail(self.dot_err(node))
            kind, sname, m = dm
            ty, ptr, arr = m[0], m[1], m[2]
            ak, an = self.agg(ty)
            if ptr == 0 and not arr and ak is not None:
                raise self.fail("cannot use %s '%s' as a value" % (ak, an))
            return 'int' if (ptr > 0 or arr) else self.norm(ty)
        if isinstance(node, Deref):
            t = self.ptr_target(node.x)
            ak, an = self.agg(t) if t is not None else (None, None)
            if ak is not None:
                raise self.fail("cannot use %s '%s' as a value" % (ak, an))
            return self.norm(t or 'int')
        if isinstance(node, Un):
            if node.op == '~' and self.rtype(node.x) == 'half':
                raise self.fail("'~' needs an integer")
            return 'int' if node.op in ('!', '~') else self.rtype(node.x)
        if isinstance(node, Bin):
            if node.op in ('&&', '||', '==', '!=', '<', '>', '<=', '>='):
                return 'int'
            lt, rt = self.rtype(node.l), self.rtype(node.r)
            if node.op in ('%', '<<', '>>', '&', '|', '^'):
                if lt == 'half' or rt == 'half':
                    raise self.fail("'%s' needs integers" % node.op)
                return 'int'
            return 'half' if (lt == 'half' or rt == 'half') else 'int'
        if isinstance(node, Cond):
            return 'half' if (self.rtype(node.t) == 'half' or
                              self.rtype(node.f) == 'half') else 'int'
        if isinstance(node, Assign):
            return self.ltype(node.target)
        if isinstance(node, Call):
            fd = self.funcs.get(node.name)
            if fd is None or fd.ret == 'void':
                raise self.fail("void function '%s' has no value" % node.name)
            return self.norm(fd.ret)
        if isinstance(node, MCall):
            key, fd = self.mcall_target(node)[1:3]
            if fd.ret == 'void':
                raise self.fail("void method '%s' has no value" % key)
            return self.norm(fd.ret)
        raise self.fail("cannot type that expression")

    def ltype(self, target):
        """type a stored value must convert to (structs fail: memcpy path)."""
        if isinstance(target, Var):
            if target.name in self.enums:
                raise self.fail("cannot assign to enumerator '%s'" % target.name)
            a = self.var_agg(target.name)
            if a is not None:
                raise self.fail(self.agg_val_err(a))
            ty, ptr, arr = self._vt(target.name)
            return 'int' if (ptr > 0 or arr) else self.norm(ty)
        if isinstance(target, Sub):
            a = self.sub_elem_struct(target.base)
            if a is not None:
                raise self.fail(self.agg_val_err(a))
            base = target.base
            if isinstance(base, Var):
                return self.norm(self._vt(base.name)[0])
            if isinstance(base, Dot):
                dm = self.dot_member(base)
                if dm is None:
                    raise self.fail(self.dot_err(base))
                return self.norm(dm[2][0])
            raise self.fail("array base must be a variable")
        if isinstance(target, Dot):
            dm = self.dot_member(target)
            if dm is None:
                raise self.fail(self.dot_err(target))
            kind, sname, m = dm
            ty, ptr, arr = m[0], m[1], m[2]
            if arr:
                raise self.fail("cannot assign to array '%s'" % target.member)
            ak, an = self.agg(ty)
            if ptr == 0 and ak is not None:
                raise self.fail("cannot use %s '%s' as a value" % (ak, an))
            return self.norm(ty)
        if isinstance(target, Deref):
            t = self.ptr_target(target.x)
            ak, an = self.agg(t) if t is not None else (None, None)
            if ak is not None:
                raise self.fail("cannot use %s '%s' as a value" % (ak, an))
            return self.norm(t or 'int')
        raise self.fail("bad assignment target")

    def conv(self, want, have):
        """convert r0 from `have` to `want` (no-op if same)."""
        if want == have:
            return
        if want == 'half':
            self.emit('itof r0, r0')
        else:
            self.emit('ftoi r0, r0')

    def gen_cond(self, node):
        """evaluate a condition; flags reflect truthiness (half-aware)."""
        self.gen(node)
        self.emit_test(self.rtype(node))

    def emit_test(self, ty):
        # -0.0 == +0.0 is true, so fcmp against +0.0 is C-correct truthiness
        self.emit('fcmp r0, 0' if ty == 'half' else 'cmp r0, 0')

    # ---- structs/unions (aggregates) ----
    @staticmethod
    def agg(ty):
        """(kind, name) for 'struct S'/'union U' typenames, else (None, None)."""
        if isinstance(ty, str) and (ty.startswith('struct ') or
                                   ty.startswith('union ')):
            return (ty.split(' ', 1)[0], ty.split(' ', 1)[1])
        return (None, None)

    @staticmethod
    def struct_name(ty):
        k, n = Gen.agg(ty)
        return n if k == 'struct' else None

    def agg_table(self, kind):
        return self.structs if kind == 'struct' else self.unions

    def member_info(self, kind, sname, mname):
        st = self.agg_table(kind).get(sname)
        if st is None:
            raise self.fail("undefined %s '%s'" % (kind, sname))
        m = st['members'].get(mname)
        if m is None:
            raise self.fail("%s '%s' has no member '%s'" % (kind, sname, mname))
        return m  # (ty, ptr, arr, off, sz)

    def agg_size(self, ty):
        kind, sname = self.agg(ty)
        if kind is None:
            return 1
        st = self.agg_table(kind).get(sname)
        if st is None:
            raise self.fail("undefined %s '%s'" % (kind, sname))
        return st['size']

    def struct_size(self, ty):
        return self.agg_size(ty)

    def var_agg(self, name):
        """(kind, name) if a variable denotes a whole aggregate value."""
        if name in self.enums:
            return None
        ty, ptr, arr = self._vt(name)
        if ptr == 0 and not arr:
            k, n = self.agg(ty)
            if k is not None:
                return (k, n)
        return None

    def var_struct(self, name):
        a = self.var_agg(name)
        return a[1] if a is not None and a[0] == 'struct' else None

    def struct_of_value(self, node):
        """(kind, name) if an lvalue denotes a whole aggregate value."""
        if isinstance(node, Var):
            return self.var_agg(node.name)
        if isinstance(node, Sub):
            return self.sub_elem_struct(node.base)
        if isinstance(node, Deref):
            t = self.ptr_target(node.x)
            if not t:
                return None
            k, n = self.agg(t)
            return (k, n) if k is not None else None
        if isinstance(node, Dot):
            return self.dot_member_struct(node)
        return None

    def sub_elem_struct(self, base):
        """(kind, name) if Sub base's elements are whole aggregates."""
        if isinstance(base, Var):
            if base.name in self.enums:
                return None
            ty, ptr, arr = self._vt(base.name)
            if arr:
                k, n = self.agg(ty)
                if k is not None:
                    return (k, n)
            return None
        if isinstance(base, Dot):
            dm = self.dot_member(base)
            if dm is not None:
                m = dm[2]
                if m[2]:
                    k, n = self.agg(m[0])
                    if k is not None:
                        return (k, n)
            return None
        return None

    def dot_member(self, node):
        """(kind, struct, member-info) for a Dot node, else None."""
        a = self.struct_of_value(node.base)
        if a is None:
            return None
        kind, sname = a
        return (kind, sname, self.member_info(kind, sname, node.member))

    def dot_err(self, node):
        op = '->' if node.arrow else '.'
        what = 'struct/union pointer' if node.arrow else 'struct or union'
        return "left of '%s%s' is not a %s" % (op, node.member, what)

    def agg_val_err(self, a):
        return "cannot use %s '%s' as a value" % (a[0], a[1])

    def dot_member_struct(self, node):
        dm = self.dot_member(node)
        if dm is None:
            return None
        kind, sname, m = dm
        ty, ptr, arr = m[0], m[1], m[2]
        if ptr == 0 and not arr:
            k, n = self.agg(ty)
            if k is not None:
                return (k, n)
        return None

    def sub_elem_words(self, base):
        """element size in words for a Sub base (Var or Dot)."""
        if isinstance(base, Var):
            if base.name in self.enums:
                raise self.fail("'%s' is not an array" % base.name)
            ty, ptr, arr = self._vt(base.name)
            if ptr > 0 or arr:
                return self.struct_size(ty)
            return 1
        if isinstance(base, Dot):
            dm = self.dot_member(base)
            if dm is None:
                raise self.fail(self.dot_err(base))
            kind, sname, m = dm
            return self.agg_size(m[0]) if m[2] else 1
        raise self.fail("array base must be a variable")

    def label(self, prefix='L'):
        self.nlabel += 1
        return '%s%d' % (prefix, self.nlabel)

    @staticmethod
    def func_label(name):
        return 'F_' + re.sub(r'\W', '_', name)

    def emit(self, s=''):
        self.out.append(s)

    def fail(self, msg):
        if self.cur_func is not None:
            msg = "in %s: %s" % (self.cur_func, msg)
        raise CompileError(msg, file=self.cur_file)

    def intern_str(self, raw):
        self.nstr += 1
        lab = 'S%d' % self.nstr
        self.data.append('%-8s .string "%s"' % (lab + ':', c_str_body(raw)))
        return lab

    # ---- lvalues ----
    def addr_of(self, node):
        """address of Var/Sub/Deref/Dot/Str into r0."""
        if isinstance(node, Var):
            if node.name in self.enums:
                raise self.fail("cannot take address of enumerator '%s'"
                                % node.name)
            self.gen_addr_var(node.name)
        elif isinstance(node, Sub):
            self.gen(node.idx)            # r0 = i
            es = self.sub_elem_words(node.base)
            if es > 1:
                self.emit('mul r0, %d' % es)
            self.emit('push r10, r0')
            if isinstance(node.base, Var) and node.base.name in self.params:
                self.gen_load_var(node.base.name)  # param holds the pointer value
            else:
                self.addr_of_base(node.base)       # array: address of slot 0
            self.emit('pop r1, r10')
            self.emit('add r0, r1')
        elif isinstance(node, Dot):
            dm = self.dot_member(node)
            if dm is None:
                raise self.fail(self.dot_err(node))
            kind, sname, m = dm
            self.addr_of_value(node.base)  # r0 = base address
            if m[3]:
                self.emit('add r0, %d' % m[3])
        elif isinstance(node, Deref):
            self.gen(node.x)              # already the address
        elif isinstance(node, Str):
            self.emit('move r0, %s' % self.intern_str(node.raw))
        else:
            raise self.fail("cannot take address of that")

    def addr_of_value(self, node):
        """address of a whole-struct base (Var/Sub/Deref/Dot) into r0."""
        if isinstance(node, (Var, Sub, Dot)):
            self.addr_of(node)
        elif isinstance(node, Deref):
            self.gen(node.x)
        else:
            raise self.fail("cannot take address of that")

    def addr_of_base(self, node):
        if isinstance(node, Var):
            self.gen_addr_var(node.name)
        elif isinstance(node, Dot):
            self.addr_of(node)
        else:
            raise self.fail("array base must be a variable")

    def gen_dot(self, node):
        """rvalue of base.member: scalars load, array members decay."""
        dm = self.dot_member(node)
        if dm is None:
            raise self.fail(self.dot_err(node))
        kind, sname, m = dm
        ty, ptr, arr = m[0], m[1], m[2]
        ak, an = self.agg(ty)
        if ptr == 0 and not arr and ak is not None:
            raise self.fail("cannot use %s '%s' as a value "
                            "(try &... or ....member)" % (ak, an))
        self.addr_of(node)
        if not arr:
            self.emit('load r0, r0')

    # ---- expressions ----
    def gen(self, node):
        if isinstance(node, Num):
            self.emit('move r0, %d' % node.v)
        elif isinstance(node, HalfLit):
            self.emit('move r0, %s' % node.text)
        elif isinstance(node, Str):
            self.emit('move r0, %s' % self.intern_str(node.raw))
        elif isinstance(node, Var):
            if node.name in self.enums and node.name not in self.locals \
                    and node.name not in self.params \
                    and node.name not in self.globals:
                self.emit('move r0, %d' % self.enums[node.name])
            else:
                a = self.var_agg(node.name)
                if a is not None:
                    raise self.fail("cannot use %s '%s' as a value "
                                    "(try &%s or %s.member)"
                                    % (a[0], a[1], node.name, node.name))
                self.gen_value_var(node.name)
        elif isinstance(node, Sizeof):
            self.emit('move r0, 1')
        elif isinstance(node, Sub):
            a = self.sub_elem_struct(node.base)
            if a is not None:
                raise self.fail("cannot use %s '%s' as a value "
                                "(try &... or ....member)" % (a[0], a[1]))
            self.addr_of(node)
            self.emit('load r0, r0')
        elif isinstance(node, Dot):
            self.gen_dot(node)
        elif isinstance(node, Deref):
            t = self.ptr_target(node.x)
            ak, an = self.agg(t) if t is not None else (None, None)
            if ak is not None:
                raise self.fail("cannot use %s '%s' as a value "
                                "(try &... or ....member)" % (ak, an))
            self.gen(node.x)
            self.emit('load r0, r0')
        elif isinstance(node, Addr):
            self.addr_of(node.x)
        elif isinstance(node, Un):
            self.gen(node.x)
            if node.op == '-':
                if self.rtype(node.x) == 'half':
                    self.emit('xor r0, 0x8000')  # flip sign bit
                else:
                    self.emit('neg r0, r0')
            elif node.op == '~':
                if self.rtype(node.x) == 'half':
                    raise self.fail("'~' needs an integer")
                self.emit('not r0, r0')
            elif node.op == '!':
                l0, l1 = self.label(), self.label()
                self.emit_test(self.rtype(node.x))
                if self.rtype(node.x) == 'half':
                    # jz (not jnz): NaN has no flags set, and !NaN == 0 in C
                    self.emit('jz %s' % l0)
                    self.emit('clr r0')
                    self.emit('jmp %s' % l1)
                    self.emit('%s:' % l0)
                    self.emit('move r0, 1')
                else:
                    self.emit('jnz %s' % l0)
                    self.emit('move r0, 1')
                    self.emit('jmp %s' % l1)
                    self.emit('%s:' % l0)
                    self.emit('clr r0')
                self.emit('%s:' % l1)
        elif isinstance(node, Bin):
            self.gen_bin(node.op, node.l, node.r)
        elif isinstance(node, Cond):
            res = self.rtype(node)
            self.gen_cond(node.c)
            lf, le = self.label(), self.label()
            self.emit('jz %s' % lf)
            self.gen(node.t)
            self.conv(res, self.rtype(node.t))
            self.emit('jmp %s' % le)
            self.emit('%s:' % lf)
            self.gen(node.f)
            self.conv(res, self.rtype(node.f))
            self.emit('%s:' % le)
        elif isinstance(node, Assign):
            self.gen_assign(node)
        elif isinstance(node, Call):
            self.gen_call(node.name, node.args, want_value=True)
        elif isinstance(node, MCall):
            self.gen_mcall(node, want_value=True)
        elif isinstance(node, Builtin):
            raise self.fail("builtins are statements, not expressions")
        else:
            raise self.fail("cannot generate that expression")

    def load_var(self, name):
        self.gen_load_var(name)

    def store_top(self, target):
        """r0 holds the value; store into Var/Sub/Deref/Dot (scalars)."""
        if isinstance(target, Var):
            if target.name in self.arrays or target.name in self.garrays:
                raise self.fail("cannot assign to array '%s'" % target.name)
            self.gen_store_var(target.name)
        elif isinstance(target, (Sub, Deref, Dot)):
            self.emit('push r10, r0')     # value
            self.addr_of(target)          # r0 = address
            self.emit('pop r1, r10')           # r1 = value... careful: pop order
            self.emit('stor r0, r1')
        else:
            raise self.fail("bad assignment target")

    def gen_bin(self, op, l, r):
        lt, rt = self.rtype(l), self.rtype(r)
        if op == '&&':
            lf, le = self.label(), self.label()
            self.gen(l)
            self.emit_test(lt)
            self.emit('jz %s' % lf)
            self.gen(r)
            self.emit_test(rt)
            self.emit('jz %s' % lf)
            self.emit('move r0, 1')
            self.emit('jmp %s' % le)
            self.emit('%s:' % lf)
            self.emit('clr r0')
            self.emit('%s:' % le)
            return
        if op == '||':
            lt2, le = self.label(), self.label()
            self.gen(l)
            self.emit_test(lt)
            self.emit('jnz %s' % lt2)
            self.gen(r)
            self.emit_test(rt)
            self.emit('jnz %s' % lt2)
            self.emit('clr r0')
            self.emit('jmp %s' % le)
            self.emit('%s:' % lt2)
            self.emit('move r0, 1')
            self.emit('%s:' % le)
            return
        if op in ('==', '!=', '<', '>', '<=', '>='):
            jmp = {'==': 'jz', '!=': 'jnz', '<': 'jn', '>': 'jp',
                   '<=': 'jle', '>=': 'jge'}[op]
            lt2, le = self.label(), self.label()
            self.gen(l)
            self.conv('half' if (lt == 'half' or rt == 'half') else 'int', lt)
            self.emit('push r10, r0')
            self.gen(r)
            self.conv('half' if (lt == 'half' or rt == 'half') else 'int', rt)
            self.emit('pop r1, r10')
            # NaN clears N/Z/P, so ==/</>/<=/>= are false and != is true
            self.emit('fcmp r1, r0' if (lt == 'half' or rt == 'half')
                      else 'cmp r1, r0')
            if (lt == 'half' or rt == 'half') and op == '!=':
                # jnz is N||P (false for NaN); branch on jz instead
                self.emit('jz %s' % lt2)
                self.emit('move r0, 1')
                self.emit('jmp %s' % le)
                self.emit('%s:' % lt2)
                self.emit('clr r0')
                self.emit('%s:' % le)
                return
            self.emit('%s %s' % (jmp, lt2))
            self.emit('clr r0')
            self.emit('jmp %s' % le)
            self.emit('%s:' % lt2)
            self.emit('move r0, 1')
            self.emit('%s:' % le)
            return
        if op in ('%', '<<', '>>', '&', '|', '^'):
            if lt == 'half' or rt == 'half':
                raise self.fail("'%s' needs integers" % op)
        if lt == 'half' or rt == 'half':
            fop = {'+': 'fadd', '-': 'fsub', '*': 'fmul', '/': 'fdiv'}[op]
            self.gen(l)
            self.conv('half', lt)
            self.emit('push r10, r0')
            self.gen(r)
            self.conv('half', rt)
            self.emit('pop r1, r10')          # r1 = left, r0 = right
            self.emit('%s r0, r1, r0' % fop)
            return
        self.gen(l)
        self.emit('push r10, r0')
        self.gen(r)
        self.emit('pop r1, r10')               # r1 = left, r0 = right
        if op == '+':
            self.emit('add r0, r1, r0')
        elif op == '-':
            self.emit('sub r0, r1, r0')
        elif op == '*':
            self.emit('mul r0, r1, r0')
        elif op == '/':
            self.emit('div r0, r1, r0')
        elif op == '%':
            self.emit('mod r0, r1, r0')
        elif op == '<<':
            self.emit('shl r0, r1, r0')
        elif op == '>>':
            self.emit('shr r0, r1, r0')
        elif op == '&':
            self.emit('and r0, r1, r0')
        elif op == '|':
            self.emit('or r0, r1, r0')
        elif op == '^':
            self.emit('xor r0, r1, r0')
        else:
            raise self.fail("unknown operator '%s'" % op)

    def gen_assign(self, node):
        op = node.op
        ts = self.struct_of_value(node.target)
        if ts is not None:
            if op != '=':
                raise self.fail("no '%s' on %ss" % (op, ts[0]))
            vs = self.struct_of_value(node.val)
            if vs is None:
                raise self.fail("cannot assign that to %s '%s'" % (ts[0], ts[1]))
            if vs != ts:
                raise self.fail("mismatched %s types ('%s' vs '%s')"
                                % (ts[0], ts[1], vs[1]))
            self.gen_struct_copy(node.target, node.val)
            return
        if op == '=':
            want = self.ltype(node.target)
            self.gen(node.val)            # r0 = value
            self.conv(want, self.rtype(node.val))
            self.store_top(node.target)
            return
        # op= : load target, apply, store. target evaluated once into r1 addr.
        if isinstance(node.target, Var):
            if node.target.name in self.arrays or node.target.name in self.garrays:
                raise self.fail("cannot assign to array '%s'" % node.target.name)
            self.gen_load_var(node.target.name)
        elif isinstance(node.target, (Sub, Deref, Dot)):
            self.addr_of(node.target)
            self.emit('load r0, r0')
        else:
            raise self.fail("bad assignment target")
        lt = self.ltype(node.target)
        rt = self.rtype(node.val)
        if op in ('%=', '<<=', '>>=', '&=', '|=', '^='):
            if lt == 'half' or rt == 'half':
                raise self.fail("'%s' needs integers" % op)
        res = 'half' if (lt == 'half' or rt == 'half') else 'int'
        self.conv(res, lt)
        self.emit('push r10, r0')         # old value
        self.gen(node.val)                # r0 = rhs
        self.conv(res, rt)
        self.emit('pop r1, r10')               # r1 = old
        if res == 'half':
            base = {'+=': 'fadd', '-=': 'fsub', '*=': 'fmul',
                    '/=': 'fdiv'}[op]
        else:
            base = {'+=': 'add', '-=': 'sub', '*=': 'mul', '/=': 'div',
                    '%=': 'mod', '&=': 'and', '|=': 'or', '^=': 'xor',
                    '<<=': 'shl', '>>=': 'shr'}[op]
        self.emit('%s r0, r1, r0' % base)
        self.conv(lt, res)
        # store r0 back
        if isinstance(node.target, Var):
            self.gen_store_var(node.target.name)
        else:
            self.emit('push r10, r0')
            self.addr_of(node.target)
            self.emit('pop r1, r10')
            self.emit('stor r0, r1')

    def gen_struct_copy(self, dst, src):
        """unrolled word copy of one whole aggregate onto another."""
        a = self.struct_of_value(dst)
        n = self.agg_table(a[0])[a[1]]['size']
        self.addr_of(src)                 # r0 = src
        self.emit('push r10, r0')
        self.addr_of(dst)                 # r0 = dst
        self.emit('move r1, r0')
        self.emit('pop r0, r10')          # r0 = src, r1 = dst
        for i in range(n):
            if i:
                self.emit('load r2, r0+%d' % i)
                self.emit('stor r1+%d, r2' % i)
            else:
                self.emit('load r2, r0')
                self.emit('stor r1, r2')

    # ---- statements ----
    def gen_stmt(self, st):
        if isinstance(st, Block):
            mark = getattr(st, '_mark', None)
            if mark is not None:
                self.emit('load r0, %s' % self.DTOP)
                self.emit('stor %s, r0' % self.mark_op(mark))
            for s in st.stmts:
                self.gen_stmt(s)
            if mark is not None:
                self.gen_unwind_to_mark(mark)
        elif isinstance(st, Defer):
            self.emit('move r0, %s' % st._label)
            self.emit('load r1, %s' % self.DTOP)
            self.emit('cmp r1, %d' % self.DMAX)
            self.emit('jge DTrap')
            self.emit('move r2, %s' % self.DSTACK)
            self.emit('add r2, r1')
            self.emit('stor r2, r0')
            self.emit('inc r1')
            self.emit('move r2, %s' % self.DTOP)
            self.emit('stor r2, r1')
        elif isinstance(st, Decl):
            self.gen_decl(st, global_=False)
        elif isinstance(st, ExprStmt):
            if isinstance(st.e, Builtin):
                self.gen_builtin(st.e)
            elif isinstance(st.e, Call):
                self.gen_call(st.e.name, st.e.args, want_value=False)
            elif isinstance(st.e, MCall):
                self.gen_mcall(st.e, want_value=False)
            else:
                self.gen(st.e)
        elif isinstance(st, If):
            lf, le = self.label(), self.label()
            self.gen_cond(st.c)
            self.emit('jz %s' % lf)
            self.gen_stmt(st.t)
            if st.f is not None:
                self.emit('jmp %s' % le)
                self.emit('%s:' % lf)
                self.gen_stmt(st.f)
                self.emit('%s:' % le)
            else:
                self.emit('%s:' % lf)
        elif isinstance(st, While):
            lt, le = self.label(), self.label()
            self.emit('%s:' % lt)
            self.gen_cond(st.c)
            self.emit('jz %s' % le)
            self.loops.append((lt, le, getattr(st.body, '_mark', None)))
            self.gen_stmt(st.body)
            self.loops.pop()
            self.emit('jmp %s' % lt)
            self.emit('%s:' % le)
        elif isinstance(st, DoWhile):
            lt = self.label()
            le = self.label()
            self.emit('%s:' % lt)
            self.loops.append((lt, le, getattr(st.body, '_mark', None)))
            self.gen_stmt(st.body)
            self.loops.pop()
            self.gen_cond(st.c)
            self.emit('jnz %s' % lt)
            self.emit('%s:' % le)
        elif isinstance(st, For):
            if isinstance(st.init, Decl):
                if st.init.init is not None and not isinstance(st.init.init, Str):
                    self.gen(st.init.init)
                    if st.init.ptr == 0 and st.init.arr is None:
                        self.conv(self.norm(st.init.typename),
                                  self.rtype(st.init.init))
                    self.gen_store_var(st.init.name)
            elif st.init is not None:
                self.gen_stmt(st.init)
            lt, le, li = self.label(), self.label(), self.label()
            self.emit('%s:' % lt)
            if st.cond is not None:
                self.gen_cond(st.cond)
                self.emit('jz %s' % le)
            self.loops.append((li, le, getattr(st.body, '_mark', None)))
            self.gen_stmt(st.body)
            self.loops.pop()
            self.emit('%s:' % li)
            if st.incr is not None:
                self.gen(st.incr)
            self.emit('jmp %s' % lt)
            self.emit('%s:' % le)
        elif isinstance(st, Return):
            if self.in_defer:
                raise self.fail("cannot return inside deferred code")
            if st.val is not None:
                if self.cur_void:
                    raise self.fail("void function should not return a value")
                self.gen(st.val)
                self.conv(self.norm(self.cur_ret), self.rtype(st.val))
            if self.func_defer:
                self.emit('push r10, r0')
                self.gen_unwind_base()
                self.emit('pop r0, r10')
                self.emit('jmp %s' % self.retend)
            else:
                self.emit('jmp %s' % self.end)
        elif isinstance(st, Break):
            if not self.loops:
                raise self.fail("break outside a loop")
            if self.loops[-1][2] is not None:
                self.gen_unwind_to_mark(self.loops[-1][2])
            self.emit('jmp %s' % self.loops[-1][1])
        elif isinstance(st, Continue):
            if not self.loops:
                raise self.fail("continue outside a loop")
            if self.loops[-1][2] is not None:
                self.gen_unwind_to_mark(self.loops[-1][2])
            self.emit('jmp %s' % self.loops[-1][0])
        else:
            raise self.fail("cannot generate that statement")

    def decl_words(self, decl):
        """total words for a variable: arrays and structs scale up."""
        if decl.ptr > 0:
            return 1
        elem = self.struct_size(decl.typename)
        return (decl.arr or 1) * elem

    def alloc_local(self, decl):
        if decl.name in self.locals or decl.name in self.globals \
                or decl.name in self.params or decl.name in self.enums:
            raise self.fail("'%s' already defined" % decl.name)
        size = self.decl_words(decl)
        self.locals[decl.name] = self.nslots
        self.local_sizes[decl.name] = size
        self.nslots += size
        if decl.arr:
            self.arrays.add(decl.name)
        self.vartype[decl.name] = (decl.typename, decl.ptr, bool(decl.arr))

    # ---- defer ----
    def alloc_hidden(self):
        """one hidden frame word (defer scope mark); returns its slot."""
        slot = self.nslots
        self.nslots += 1
        return slot

    def mark_op(self, slot):
        return 'r11-%d' % (slot + 1)

    def mark_scopes(self, st, is_root=False):
        """True if the subtree contains a Defer. Annotates Blocks that do
        with _mark slots (except a defer-body's own top Block, and, when
        is_root, the function body itself). Appends Defer sites found."""
        if isinstance(st, Defer):
            self.defer_sites.append(st)
            sub = st.stmt.stmts if isinstance(st.stmt, Block) else [st.stmt]
            for s in sub:
                self.mark_scopes(s)
            return True
        if isinstance(st, Block):
            found = False
            for s in st.stmts:
                if self.mark_scopes(s):
                    found = True
            if found and not is_root:
                st._mark = self.alloc_hidden()
            return found
        if isinstance(st, Decl):
            return False
        if isinstance(st, If):
            a = self.mark_scopes(st.t)
            b = self.mark_scopes(st.f) if st.f is not None else False
            return a or b
        if isinstance(st, (While, DoWhile)):
            return self.mark_scopes(st.body)
        if isinstance(st, For):
            return self.mark_scopes(st.body)
        return False

    def plan_defer_body(self, site):
        """assign a label, collect scratch locals, snapshot function context."""
        site._label = 'D%d' % self.defer_seq
        self.defer_seq += 1
        loc, sizes, vts, arrs = {}, {}, {}, set()
        total = [0]
        self.collect_scratch(site.stmt, loc, sizes, vts, arrs, total)
        site._locals, site._sizes = loc, sizes
        site._vartype, site._arrays = vts, arrs
        site._ctx = (dict(self.locals), dict(self.local_sizes),
                     dict(self.vartype), set(self.arrays), dict(self.params),
                     self.cur_file, self.cur_func)
        self.deferred.append(site)
        if total[0] > self.scratch_size:
            self.scratch_size = total[0]

    def collect_scratch(self, st, loc, sizes, vts, arrs, total):
        """collect one deferred body's locals into scratch slots (absolute).
        Shadowing outer names is allowed; duplicates within the body fail.
        Nested Defer subtrees are skipped (planned separately)."""
        if isinstance(st, Block):
            for s in st.stmts:
                self.collect_scratch(s, loc, sizes, vts, arrs, total)
        elif isinstance(st, Decl):
            if st.name in loc:
                raise self.fail("'%s' already defined" % st.name)
            size = self.decl_words(st)
            slot = self.scratch_base + total[0]
            loc[st.name] = slot
            sizes[st.name] = size
            total[0] += size
            if st.arr:
                arrs.add(st.name)
            vts[st.name] = (st.typename, st.ptr, bool(st.arr))
        elif isinstance(st, Defer):
            pass
        elif isinstance(st, If):
            self.collect_scratch(st.t, loc, sizes, vts, arrs, total)
            if st.f is not None:
                self.collect_scratch(st.f, loc, sizes, vts, arrs, total)
        elif isinstance(st, (While, DoWhile)):
            self.collect_scratch(st.body, loc, sizes, vts, arrs, total)
        elif isinstance(st, For):
            if isinstance(st.init, Decl):
                self.collect_scratch(st.init, loc, sizes, vts, arrs, total)
            self.collect_scratch(st.body, loc, sizes, vts, arrs, total)

    def gen_unwind_to_mark(self, slot):
        # reload the mark inside the loop: called bodies clobber r0-r2
        lu, ld = self.label('U'), self.label('U')
        self.emit('%s:' % lu)
        self.emit('load r0, %s' % self.mark_op(slot))
        self.emit('load r1, %s' % self.DTOP)
        self.emit('cmp r1, r0')
        self.emit('jz %s' % ld)
        self.emit('dec r1')
        self.emit('move r2, %s' % self.DTOP)
        self.emit('stor r2, r1')
        self.emit('move r2, %s' % self.DSTACK)
        self.emit('add r2, r1')
        self.emit('load r2, r2')
        self.emit('call r10, r2')
        self.emit('jmp %s' % lu)
        self.emit('%s:' % ld)

    def gen_unwind_base(self):
        self.gen_unwind_to_mark(self.base_mark)

    def gen_decl(self, decl, global_):
        if global_:
            if decl.name in self.globals or decl.name in self.enums:
                raise self.fail("'%s' already defined" % decl.name)
            lab = 'G_' + decl.name
            size = self.decl_words(decl)
            if isinstance(decl.init, Str):
                if decl.arr is not None:
                    raise self.fail("array cannot init from string here")
                self.data.append('%-8s .word %s' % (lab + ':', lab + '_s'))
                self.data.append('%-8s .string "%s"' % (lab + '_s:', c_str_body(decl.init.raw)))
            elif decl.init is not None:
                if decl.arr is not None:
                    raise self.fail("array initializers need M2")
                ak, an = self.agg(decl.typename)
                if decl.ptr == 0 and ak is not None:
                    raise self.fail("cannot initialize %ss" % ak)
                v, have = self.const_val(decl.init)
                want = self.norm(decl.typename) if decl.ptr == 0 else 'int'
                self.data.append('%-8s .word %s' % (lab + ':',
                                                    self.const_conv(want, have, v)))
            else:
                self.data.append('%-8s .fill %d' % (lab + ':', size))
            self.globals[decl.name] = (lab, size)
            self.gvartype[decl.name] = (decl.typename, decl.ptr,
                                        decl.arr is not None)
            if decl.arr is not None:
                self.garrays.add(decl.name)
        else:
            if decl.name not in self.locals:
                raise self.fail("internal: '%s' was not pre-collected" % decl.name)
            if decl.init is not None:
                if isinstance(decl.init, Str):
                    if decl.arr is not None or decl.ptr == 0:
                        raise self.fail("string init needs a char* variable")
                    lab = self.intern_str(decl.init.raw)
                    self.emit('move r0, %s' % lab)
                    self.gen_store_var(decl.name)
                    return
                if decl.arr is not None:
                    raise self.fail("array initializers need M2")
                ak, an = self.agg(decl.typename)
                if decl.ptr == 0 and ak is not None:
                    s = self.struct_of_value(decl.init)
                    if s != (ak, an):
                        raise self.fail("cannot initialize %s '%s' from that"
                                        % (ak, an))
                    self.gen_struct_copy(Var(decl.name), decl.init)
                    return
                self.gen(decl.init)
                if decl.ptr == 0:
                    self.conv(self.norm(decl.typename),
                              self.rtype(decl.init))
                self.gen_store_var(decl.name)

    def const_val(self, node):
        """constant initializer -> (text, kind). Folds ints (enums, sizeof)."""
        if isinstance(node, Num):
            return (str(node.v), 'int')
        if isinstance(node, HalfLit):
            return (node.text, 'half')
        if isinstance(node, Var):
            if node.name in self.enums:
                return (str(self.enums[node.name]), 'int')
            raise self.fail("global initializer must be a constant")
        if isinstance(node, Sizeof):
            return ('1', 'int')
        if isinstance(node, Un) and node.op == '-':
            t, k = self.const_val(node.x)
            if k == 'half':
                return ('-' + t, 'half')
            return (str(self._neg(t)), 'int')
        if isinstance(node, (Bin, Un)):
            return (str(self.fold_node(node)), 'int')
        raise self.fail("global initializer must be a constant")

    def _neg(self, text):
        return (-int(text, 0)) & 0xFFFF

    def fold_node(self, node):
        """fold an int constant expression (mirrors Parser.fold_int)."""
        if isinstance(node, Num):
            return node.v & 0xFFFF
        if isinstance(node, Sizeof):
            return 1
        if isinstance(node, Var):
            if node.name in self.enums:
                return self.enums[node.name]
            raise self.fail("'%s' is not a constant" % node.name)
        if isinstance(node, Un) and node.op == '-':
            return (-self.fold_node(node.x)) & 0xFFFF
        if isinstance(node, Un) and node.op == '~':
            return (~self.fold_node(node.x)) & 0xFFFF
        if isinstance(node, Un) and node.op == '+':
            return self.fold_node(node.x)
        if isinstance(node, Un) and node.op == '!':
            return 0 if self.fold_node(node.x) else 1
        if isinstance(node, Bin):
            a, b = self.fold_node(node.l), self.fold_node(node.r)
            sa = lambda v: v - 65536 if v >= 32768 else v
            op = node.op
            if op == '+':
                return (a + b) & 0xFFFF
            if op == '-':
                return (a - b) & 0xFFFF
            if op == '*':
                return (a * b) & 0xFFFF
            if op == '/':
                if not b:
                    raise self.fail("division by zero")
                return int(sa(a) / sa(b)) & 0xFFFF
            if op == '%':
                if not b:
                    raise self.fail("division by zero")
                q = abs(sa(a)) // abs(sa(b))
                r = abs(sa(a)) - abs(sa(b)) * q
                return ((r if sa(a) >= 0 else -r) & 0xFFFF)
            if op == '<<':
                return (a << b) & 0xFFFF
            if op == '>>':
                return (sa(a) >> b) & 0xFFFF
            if op == '&':
                return a & b
            if op == '|':
                return a | b
            if op == '^':
                return a ^ b
            if op in ('==', '!=', '<', '>', '<=', '>='):
                av, bv = sa(a), sa(b)
                r = {'==': av == bv, '!=': av != bv, '<': av < bv,
                     '>': av > bv, '<=': av <= bv, '>=': av >= bv}[op]
                return 1 if r else 0
            if op == '&&':
                return 1 if (a and b) else 0
            if op == '||':
                return 1 if (a or b) else 0
        raise self.fail("global initializer must be a constant")

    def const_conv(self, want, have, text):
        """fold an int<->half conversion into assembler expression text."""
        if want == have:
            return text
        if want == 'half':
            try:
                n = int(text, 0)
            except ValueError:
                raise self.fail("bad integer constant '%s'" % text)
            if n >= 32768:  # folded words are unsigned; interpret as signed
                n -= 65536
            return '%d.0h' % n
        # half -> int truncates (C-like); NaN/Inf cannot initialize an int
        try:
            f = float(text[:-1] if text.endswith('h') else text)
        except ValueError:
            raise self.fail("cannot init int from '%s'" % text)
        if f != f or f > 32767.0 or f < -32768.0:
            raise self.fail("cannot init int from '%s'" % text)
        return str(int(f))

    # ---- builtins (inline) ----
    def gen_builtin(self, b):
        if b.name == 'puts':
            # null-terminated word string + newline, pointer in r0
            if len(b.args) != 1:
                raise self.fail("puts takes 1 argument")
            a = b.args[0]
            if isinstance(a, Str):
                lab = self.intern_str(a.raw)
                self.emit('move r0, %s' % lab)
            else:
                if self.rtype(a) == 'half':
                    raise self.fail("puts needs a string, not half")
                self.gen(a)
            lp, le = self.label('P'), self.label('P')
            self.emit('%s:' % lp)
            self.emit('load r1, r0')
            self.emit('cbz r1, %s' % le)
            self.emit('clr r2')
            self.emit('out r2, r1')
            self.emit('inc r0')
            self.emit('jmp %s' % lp)
            self.emit('%s:' % le)
            self.emit('clr r2')
            self.emit('move r1, 10')
            self.emit('out r2, r1')
            return
        if b.name == 'putc':
            # value in r0, port 0 in r1
            if len(b.args) != 1:
                raise self.fail("putc takes 1 argument")
            self.gen(b.args[0])
            self.conv('int', self.rtype(b.args[0]))
            self.emit('clr r1')
            self.emit('out r1, r0')
            return
        if b.name == 'print_int':
            # signed 16-bit decimal + no newline; temps r1-r5
            if len(b.args) != 1:
                raise self.fail("print_int takes 1 argument")
            self.gen(b.args[0])
            self.conv('int', self.rtype(b.args[0]))
            ln, lq, lo, ld = (self.label('P') for _ in range(4))
            self.emit('cmp r0, 0')
            self.emit('jge %s' % ln)
            self.emit('clr r1')
            self.emit('move r2, 45')
            self.emit('out r1, r2')
            self.emit('neg r0, r0')
            self.emit('%s:' % ln)
            self.emit('move r2, r0')
            self.emit('move r3, 10')
            self.emit('clr r4')
            self.emit('%s:' % lq)
            self.emit('divmod r5, r2, r3')
            self.emit('push r10, r2')
            self.emit('inc r4')
            self.emit('move r2, r5')
            self.emit('cmp r2, 0')
            self.emit('jnz %s' % lq)
            self.emit('%s:' % lo)
            self.emit('cmp r4, 0')
            self.emit('jz %s' % ld)
            self.emit('pop r2, r10')
            self.emit('add r2, 48')
            self.emit('clr r1')
            self.emit('out r1, r2')
            self.emit('dec r4')
            self.emit('jmp %s' % lo)
            self.emit('%s:' % ld)
            return
        if b.name == 'gettime':
            # 64-bit virtual time -> four words at the address in r0
            if len(b.args) != 1:
                raise self.fail("gettime takes 1 argument")
            self.gen_time_addr(b)
            for i in range(4):
                self.emit('in r1, 0x%X' % (0x70 + i))
                self.emit('stor r0+%d, r1' % i if i else 'stor r0, r1')
            return
        if b.name == 'settime':
            # four words at the address in r0 -> 64-bit virtual time
            if len(b.args) != 1:
                raise self.fail("settime takes 1 argument")
            self.gen_time_addr(b)
            self.emit('move r2, 0x70')
            for i in range(4):
                self.emit('load r1, r0+%d' % i if i else 'load r1, r0')
                self.emit('out r2, r1')
                if i < 3:
                    self.emit('inc r2')
            return
        raise self.fail("unknown builtin '%s'" % b.name)

    def gen_time_addr(self, b):
        """evaluate a gettime/settime address argument into r0."""
        a = b.args[0]
        if isinstance(a, Str):
            lab = self.intern_str(a.raw)
            self.emit('move r0, %s' % lab)
        else:
            if self.rtype(a) == 'half':
                raise self.fail("%s needs an address, not half" % b.name)
            self.gen(a)

    # ---- program ----
    def generate(self, globals_, funcs, main):
        for d in globals_:
            self.cur_file, self.cur_func = d.src or self.cur_file, None
            self.gen_decl(d, global_=True)
        # collect every function first (forward calls, mutual recursion)
        for fd in funcs:
            if fd.name in self.funcs:
                raise self.fail("duplicate function '%s'" % fd.name)
            self.funcs[fd.name] = fd
        self.emit('enter2 main2')
        self.emit('halt')
        self.emit('.mode 2')
        # main first: enter2 above is a 9-bit pc-relative jump, so the boot
        # target must stay close no matter how big other functions get
        for fd in funcs:
            if fd.name == 'main':
                self.gen_func(fd)
        for fd in funcs:
            if fd.name != 'main':
                self.gen_func(fd)
        if self.need_defer:
            self.gen_defer_runtime()
            self.gen_deferred_bodies()
        for d in self.data:
            self.emit(d)

    def gen_defer_runtime(self):
        for gname in ('defer_stack', 'defer_top'):
            if gname in self.globals:
                raise self.fail("'%s' is already defined" % gname)
        self.data.append('%-8s .fill %d' % ('G_defer_stack:', self.DMAX))
        self.data.append('%-8s .word 0' % 'G_defer_top:')
        self.emit('DTrap:')
        self.emit('clr r0')
        self.emit('div r0, r0, r0')  # div by zero: overflow trap

    def gen_deferred_bodies(self):
        for site in self.deferred:
            (floc, fsizes, fvts, farrs, fparams,
             ffile, ffunc) = site._ctx
            save = (self.locals, self.local_sizes, self.vartype,
                    self.arrays, self.params, self.cur_file, self.cur_func,
                    self.in_defer)
            merged_locals = dict(floc)
            merged_locals.update(site._locals)
            merged_sizes = dict(fsizes)
            merged_sizes.update(site._sizes)
            merged_vts = dict(fvts)
            merged_vts.update(site._vartype)
            merged_arrs = set(farrs) | site._arrays
            self.locals, self.local_sizes = merged_locals, merged_sizes
            self.vartype, self.arrays = merged_vts, merged_arrs
            self.params = fparams
            self.cur_file, self.cur_func = ffile, ffunc
            self.in_defer += 1
            self.loops = []
            self.emit('%s:' % site._label)
            self.gen_stmt(site.stmt)
            self.emit('ret r10')
            (self.locals, self.local_sizes, self.vartype,
             self.arrays, self.params, self.cur_file, self.cur_func,
             self.in_defer) = save

    def gen_func(self, fd):
        # fresh frame: params live at fp+(2+i), locals below fp
        self.locals, self.nslots, self.params, self.arrays = {}, 0, {}, set()
        self.local_sizes, self.vartype = {}, {}
        for i, p in enumerate(fd.params):
            if p.name in self.params:
                raise self.fail("duplicate parameter '%s'" % p.name)
            self.params[p.name] = i
            self.vartype[p.name] = (p.typename, p.ptr, False)
        self.cur_void = (fd.ret == 'void')
        self.cur_ret = fd.ret
        self.cur_file = fd.src or self.cur_file
        self.cur_func = fd.name
        self.collect_locals(fd.body)
        # defer pre-pass: scope marks, body labels/scratch (frame grows)
        self.defer_sites = []
        self.base_mark = None
        self.scratch_base = 0
        self.scratch_size = 0
        self.func_defer = self.mark_scopes(fd.body, is_root=True)
        if self.func_defer:
            self.need_defer = True
            self.base_mark = self.alloc_hidden()
            self.scratch_base = self.nslots
            for site in self.defer_sites:
                self.plan_defer_body(site)
            self.nslots += self.scratch_size
        self.loops = []
        self.end = 'Lend_' + re.sub(r'\W', '_', fd.name)
        self.retend = 'Lret_' + re.sub(r'\W', '_', fd.name)
        n = self.nslots
        if fd.name == 'main':
            self.emit('main2:')
            self.emit('move r10, 0xFFFF')
        else:
            lab = self.func_label(fd.name)
            if lab in self.func_labels and self.func_labels[lab] != fd.name:
                raise self.fail("method label collision for '%s'" % fd.name)
            self.func_labels[lab] = fd.name
            self.emit(lab + ':')
        self.emit('enter r11, r10, %d' % n)
        if self.func_defer:
            self.emit('load r0, %s' % self.DTOP)
            self.emit('stor %s, r0' % self.mark_op(self.base_mark))
        self.gen_stmt(fd.body)
        if self.func_defer:
            self.emit('%s:' % self.end)
            self.gen_unwind_base()
            self.emit('%s:' % self.retend)
        else:
            self.emit('%s:' % self.end)
        self.emit('leave r11, r10')
        self.emit('halt' if fd.name == 'main' else 'ret r10')

    def gen_call(self, name, args, want_value):
        if name in ('puts', 'putc', 'print_int', 'gettime', 'settime'):
            raise self.fail("'%s' is a statement-only builtin in M2" % name)
        if name not in self.funcs:
            raise self.fail("undefined function '%s'" % name)
        fd = self.funcs[name]
        if len(args) != len(fd.params):
            raise self.fail("'%s' takes %d argument(s), got %d"
                               % (name, len(fd.params), len(args)))
        if fd.ret == 'void' and want_value:
            raise self.fail("void function '%s' has no value" % name)
        for i in reversed(range(len(args))):
            self.gen_push_arg(name, fd.params[i], args[i])
        self.emit('call r10, %s' % self.func_label(name))
        if args:
            self.emit('add r10, r10, %d' % len(args))

    def gen_push_arg(self, fname, p, a):
        """evaluate one argument into r0 and push it (with conversions)."""
        pk, pn = self.agg(p.typename)
        if pk is not None and p.ptr > 0:
            s = self.struct_of_value(a)
            if s is None and isinstance(a, Addr):
                s = self.struct_of_value(a.x)
                if s is not None:
                    self.gen(a)
                    self.emit('push r10, r0')
                    return
            if s is None:
                if isinstance(a, Var):
                    try:
                        ty, ptr, arr = self._vt(a.name)
                    except CompileError:
                        ty, ptr = None, 0
                    ak, an = self.agg(ty) if ty is not None else (None, None)
                    if (ak, an) == (pk, pn) and ptr > 0:
                        self.gen(a)
                        self.emit('push r10, r0')
                        return
                raise self.fail("'%s' needs &%s '%s'" % (fname, pk, pn))
            if s != (pk, pn):
                raise self.fail("mismatched %s types ('%s' vs '%s')"
                                % (pk, pn, s[1]))
            self.addr_of(a)
        else:
            self.gen(a)
            if p.ptr == 0 and p.arr is None:
                self.conv(self.norm(p.typename), self.rtype(a))
        self.emit('push r10, r0')

    def mcall_target(self, node):
        """resolve base.method -> ((kind, sname), key, fd, this_is_ptr)."""
        base = node.base
        if node.arrow:
            # parser wraps `p->` bases as Deref(p); the pointer is inside
            px = base.x if isinstance(base, Deref) else base
            t = self.ptr_target(px)
            ak, an = self.agg(t) if t is not None else (None, None)
            if ak is None:
                raise self.fail("left of '->%s' is not a struct/union pointer"
                                % node.method)
            kind, sname, this_is_ptr = ak, an, True
        else:
            a = self.struct_of_value(base)
            if a is None:
                raise self.fail("left of '.%s' is not a struct or union"
                                % node.method)
            kind, sname, this_is_ptr = a[0], a[1], False
        key = sname + '.' + node.method
        fd = self.funcs.get(key)
        if fd is None:
            raise self.fail("%s '%s' has no method '%s'"
                            % (kind, sname, node.method))
        return ((kind, sname), key, fd, this_is_ptr)

    def gen_mcall(self, node, want_value):
        """`base.method(args)`: this (= &base or base pointer) is arg 0."""
        (kind, sname), key, fd, this_is_ptr = self.mcall_target(node)
        if fd.ret == 'void' and want_value:
            raise self.fail("void method '%s' has no value" % key)
        if len(node.args) != len(fd.params) - 1:
            raise self.fail("'%s' takes %d argument(s), got %d"
                            % (key, len(fd.params) - 1, len(node.args)))
        for i in reversed(range(len(node.args))):
            self.gen_push_arg(key, fd.params[i + 1], node.args[i])
        if this_is_ptr:
            px = node.base.x if isinstance(node.base, Deref) else node.base
            self.gen(px)
        else:
            self.addr_of(node.base)
        self.emit('push r10, r0')
        self.emit('call r10, %s' % self.func_label(key))
        self.emit('add r10, r10, %d' % (len(node.args) + 1))

    def collect_locals(self, st):
        if isinstance(st, Block):
            for s in st.stmts:
                self.collect_locals(s)
        elif isinstance(st, Decl):
            self.alloc_local(st)
        elif isinstance(st, Defer):
            pass  # deferred bodies use the shared scratch region (see below)
        elif isinstance(st, If):
            self.collect_locals(st.t)
            if st.f is not None:
                self.collect_locals(st.f)
        elif isinstance(st, While):
            self.collect_locals(st.body)
        elif isinstance(st, DoWhile):
            self.collect_locals(st.body)
        elif isinstance(st, For):
            if isinstance(st.init, Decl):
                self.alloc_local(st.init)
            self.collect_locals(st.body)


def compile_unit(path):
    """compile a file plus its #includes into assembler text."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError as e:
        raise CompileError(str(e), file=path)
    toks = lex(text, path, os.path.dirname(os.path.abspath(path)))
    p = Parser(toks)
    p.seen.add(os.path.realpath(path))
    globals_, funcs, main = p.parse_program()
    g = Gen()
    g.cur_file = path
    g.structs, g.unions, g.enums = p.structs, p.unions, p.enums
    g.generate(globals_, funcs, main)
    return '\n'.join(g.out) + '\n'


def compile_text(text, fname='<string>', basedir='.'):
    toks = lex(text, fname, basedir)
    p = Parser(toks, basedir)
    globals_, funcs, main = p.parse_program()
    g = Gen()
    g.cur_file = fname
    g.structs, g.unions, g.enums = p.structs, p.unions, p.enums
    g.generate(globals_, funcs, main)
    return '\n'.join(g.out) + '\n'


def main():
    args = sys.argv[1:]
    if len(args) < 1 or len(args) > 3:
        sys.exit("usage: bobc.py prog.b [-o prog.asm]")
    src = args[0]
    out = None
    if '-o' in args:
        i = args.index('-o')
        try:
            out = args[i + 1]
        except IndexError:
            sys.exit("bobc: -o needs a file")
    try:
        asm = compile_unit(src)
    except CompileError as e:
        if e.file is not None and e.line is not None:
            sys.exit("bobc: %s:%d: error: %s" % (e.file, e.line, e))
        elif e.file is not None:
            sys.exit("bobc: %s: error: %s" % (e.file, e))
        else:
            sys.exit("bobc: error: %s" % e)
    if out is None:
        out = src.rsplit('.', 1)[0] + '.asm'
    with open(out, 'w') as f:
        f.write(asm)
    print("%s -> %s" % (src, out))


if __name__ == '__main__':
    main()
