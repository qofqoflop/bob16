/*
 * BOB-16 emulator: loads a raw binary and runs it.
 *
 * Usage: ./bob16 program.bin
 *
 * File format: raw 16-bit words, LITTLE-endian, no header.
 * Word 0 of the file is loaded at address 0x0000 and pc starts at 0.
 * (For big-endian files, swap the two bytes in load_program.)
 *
 * Instruction encoding follows the original bob16 emulator decoder.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MEM_WORDS 0x10000

typedef uint16_t word;

enum { NOP, ADD, AND, NOT, LD, LDI, LDR, ST, STI, STR,
       BR, JMP, JSR, LEA, RET, TRAP };
enum { T_HALT, T_PUTC, T_PUTS, T_GETS, T_EXT };
enum { N, Z, P, C };
enum { I2_NOP, I2_HALT, I2_UEXT, I2_ADD,
       I2_SUB, I2_MUL, I2_AND, I2_OR,
       I2_NOT, I2_XOR, I2_JMP, I2_JN,
       I2_JZ, I2_JP, I2_LOAD, I2_LOAD2,
       I2_STOR, I2_STOR2, I2_IN, I2_OUT,
       I2_PUSH, I2_POP, I2_MOVE, I2_SWAP,
       I2_CALL, I2_RET, I2_ID, I2_SHL,
       I2_SHR, I2_SAR, I2_DIV, I2_MOD,
       I2_CMP, I2_JNZ, I2_JLE, I2_JGE,
       I2_SDIV, I2_SMOD, I2_ADC, I2_SBB,
       I2_JC, I2_TST, I2_JNC, I2_NEG,
       I2_MULH, I2_MULHS, I2_DIVMOD, I2_ROL,
       I2_ROR, I2_RCR, I2_BSET, I2_BCLR,
       I2_BTST, I2_LDB, I2_STB, I2_PEEK,
       I2_PUSHM, I2_POPM, I2_JAL };

/* id selectors (cpuid-like): `id rd, sel` puts the answer for `sel` in rd.
 * Unknown selectors return 0; ID_MAX returns the highest valid selector. */
enum { ID_MAX,        /* highest valid selector */
       ID_VERSION,    /* (major << 8) | minor */
       ID_FEATURES,   /* FEAT_* bit mask */
       ID_MEMTOP,     /* highest valid memory address */
       ID_RS0,        /* status register rs0 (bit 0 = extended mode) */
       ID_REGS,       /* (mode-1 register count << 8) | mode-2 register count */
       ID_OPCODES,    /* number of mode-2 opcodes */
       ID_PORTS,      /* bit n set = in/out port n exists */
       ID_NAME0,      /* name string "BOB-16", two chars per selector, */
       ID_NAME1,      /*   first char in the high byte */
       ID_NAME2 };

enum { FEAT_MEMOPS = 1 << 0,   /* [reg] memory operands on ALU instructions */
       FEAT_IMM16  = 1 << 1,   /* i3: 16-bit immediate third word */
       FEAT_SDIV   = 1 << 2,   /* signed sdiv / smod */
       FEAT_IO     = 1 << 3,   /* in / out */
       FEAT_STACK  = 1 << 4,   /* push / pop / call / ret */
       FEAT_OFF    = 1 << 5,   /* i4/i5: 8-bit offsets on src/dst */
       FEAT_EXTALU = 1 << 6,   /* tst/neg/mulh/mulhs/divmod/rol/ror/rcr/bset/bclr/btst */
       FEAT_MEMB   = 1 << 7,   /* ldb/stb byte memory */
       FEAT_STACK2 = 1 << 8,   /* peek/pushm/popm */
       FEAT_JX     = 1 << 9 }; /* jnc/jal */

enum { PORTF_IO   = 1 << 0,
       PORTF_TIME = 1 << 1 };

#define BOB16_VERSION 0x0102   /* 1.2: extended mode-2 set */

static struct {
    word r[16];
    word ir;
    word pc;
    word rs0;
    bool cc[4];
} cpu;

static word mem[MEM_WORDS];
static bool running = true;
static int64_t virtual_time;
static int64_t actual_time;

/* sign-extend the low `bits` bits of val */
static int sext(word val, int bits) {
    int shift = 16 - bits;
    return (int16_t)(word)(val << shift) >> shift;
}

static void set_cc(word val) {
    int16_t s = (int16_t)val;
    cpu.cc[N] = s < 0;
    cpu.cc[Z] = s == 0;
    cpu.cc[P] = s > 0;
}

static void bad(word addr) {
    fflush(stdout);
    fprintf(stderr, "bad instruction 0x%04X at address 0x%04X\n", cpu.ir, addr);
    exit(1);
}

static void load_program(const char *path) {
    static unsigned char buf[MEM_WORDS * 2 + 1];

    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(1);
    }
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);

    if (n > MEM_WORDS * 2) {
        fprintf(stderr, "program too large (max %d words)\n", MEM_WORDS);
        exit(1);
    }
    if (n % 2) {
        fprintf(stderr, "file size must be a multiple of 2 bytes\n");
        exit(1);
    }
    for (size_t i = 0; i < n / 2; i++) {
        mem[i] = (word)(buf[2 * i] | (buf[2 * i + 1] << 8));
    }
}

static int64_t gettime(void) {
    return virtual_time + ((int64_t)time(NULL) - actual_time);
}

static void settime(int64_t t) {
    virtual_time = t;
    actual_time = time(NULL);
}

static void step1(void) {
    word at = cpu.pc;
    cpu.ir = mem[cpu.pc++];   /* pc is incremented before execution */
    word ir = cpu.ir;
    int op = ir >> 12;
    int dr = (ir >> 9) & 7;   /* dst (or src for stores) */
    word addr, res;

    switch (op) {
    case NOP:
        break;

    case ADD:
    case AND: {
        int a, b;
        switch ((ir >> 7) & 3) {
        case 0:  /* op rd rs1 rs2 */
            if (ir & 1) bad(at);
            a = cpu.r[(ir >> 4) & 7];
            b = cpu.r[(ir >> 1) & 7];
            break;
        case 1:  /* op rd rs1 imm4 */
            a = cpu.r[(ir >> 4) & 7];
            b = sext(ir & 0xF, 4);
            break;
        case 2:  /* op rd rs */
            if (ir & 0xF) bad(at);
            a = cpu.r[dr];
            b = cpu.r[(ir >> 4) & 7];
            break;
        default: /* op rd imm7 */
            a = cpu.r[dr];
            b = sext(ir & 0x7F, 7);
            break;
        }
        res = (word)(op == ADD ? a + b : a & b);
        cpu.r[dr] = res;
        set_cc(res);
        break;
    }

    case NOT:
        if (ir & 0x1F) bad(at);
        if (ir & 0x100)
            res = (word)~cpu.r[dr];             /* not rd */
        else
            res = (word)~cpu.r[(ir >> 5) & 7];  /* not rd rs */
        cpu.r[dr] = res;
        set_cc(res);
        break;

    case LD:
        addr = (word)(cpu.pc + sext(ir & 0x1FF, 9));
        cpu.r[dr] = mem[addr];
        set_cc(cpu.r[dr]);
        break;

    case LDI:
        addr = (word)(cpu.pc + sext(ir & 0x1FF, 9));
        cpu.r[dr] = mem[mem[addr]];
        set_cc(cpu.r[dr]);
        break;

    case LDR:
        addr = (word)(cpu.r[(ir >> 6) & 7] + sext(ir & 0x3F, 6));
        cpu.r[dr] = mem[addr];
        set_cc(cpu.r[dr]);
        break;

    case ST:
        addr = (word)(cpu.pc + sext(ir & 0x1FF, 9));
        mem[addr] = cpu.r[dr];
        break;

    case STI:
        addr = (word)(cpu.pc + sext(ir & 0x1FF, 9));
        mem[mem[addr]] = cpu.r[dr];
        break;

    case STR:
        addr = (word)(cpu.r[(ir >> 6) & 7] + sext(ir & 0x3F, 6));
        mem[addr] = cpu.r[dr];
        break;

    case BR:
        if ((cpu.cc[N] && (ir & 0x800)) ||
            (cpu.cc[Z] && (ir & 0x400)) ||
            (cpu.cc[P] && (ir & 0x200))) {
            cpu.pc = (word)(cpu.pc + sext(ir & 0x1FF, 9));
        }
        break;

    case JMP:
        if (ir & 0x1FF) bad(at);
        cpu.pc = cpu.r[dr];
        break;

    case JSR: {
        /* same encoding as the original bob16: bit 11 clear = jsr imm11
         * (signed, bits 10:0), bit 11 set = jsrr rs (rs in bits 10:8).
         * The target is read before r7 is overwritten, so jsrr r7 works. */
        word target = (ir & 0x800) ? cpu.r[(ir >> 8) & 7]
                                   : (word)(cpu.pc + sext(ir & 0x7FF, 11));
        cpu.r[7] = cpu.pc;
        cpu.pc = target;
        break;
    }

    case LEA:
        res = (word)(cpu.pc + sext(ir & 0x1FF, 9));
        cpu.r[dr] = res;
        set_cc(res);
        break;

    case RET:
        cpu.pc = cpu.r[7];
        break;

    case TRAP:
        if (ir & 0xFF) bad(at);
        switch ((ir >> 8) & 0xF) {
        case T_HALT:
            running = false;
            break;

        case T_PUTC:
            putchar(cpu.r[0] & 0xFF);
            break;

        case T_PUTS: {
            cpu.r[7] = cpu.pc;
            for (word a = cpu.r[0]; mem[a] != 0; a++) {
                putchar(mem[a] & 0xFF);
            }
            putchar('\n');
            cpu.r[0] = 0;   /* the original leaves the terminator (0) in r0 */
            break;
        }

        case T_GETS: {
            static char buf[MEM_WORDS + 1];
            int n = cpu.r[1];
            cpu.r[7] = cpu.pc;
            fflush(stdout);
            if (n > 0 && fgets(buf, n, stdin)) {
                size_t len = strlen(buf);
                for (size_t i = 0; i < len; i++) {
                    mem[(word)(cpu.r[0] + i)] = (unsigned char)buf[i];
                }
                mem[(word)(cpu.r[0] + len)] = 0;
            }
            break;
        }

        case T_EXT: {
            cpu.rs0 = cpu.rs0 | 0x1;
            cpu.pc = cpu.r[7];
            break;
        }

        default:        /* vectors 5-15 are no-ops, as in the original */
            break;
        }
        break;
    }
}

#define alu_src word a, b, tmp, tmp2; \
                a = cpu.r[or1]; \
                if (i4) { \
                    tmp = mem[i3 ? at3 : at2] & 0xFF; \
                    if (i5) a -= tmp; \
                    else a += tmp; \
                } \
                b = i2 ? or2 : i3 ? mem[at2] : cpu.r[or2]; \
                if (imem1) a = mem[a]; \
                if (imem2) b = mem[b]

#define alu_pre word r; alu_src

/* With i3 set, the third word replaces the or1 operand of non-ALU
 * instructions (move/load/stor/in/out/push/call/id).
 * With i4 set, low offset byte adjusts or1 (src) and high adjusts dr (dst),
 * sharing sign i5. ALU uses alu_src/alu_post; others use or1v/droff/jmpv. */
#define or1vb (i3 ? mem[at2] : cpu.r[or1])
#define or1v (i4 ? (i5 ? (word)(or1vb - soff) : (word)(or1vb + soff)) : or1vb)
#define droff (i4 ? (i5 ? (word)(cpu.r[dr] - doff) : (word)(cpu.r[dr] + doff)) : cpu.r[dr])
#define jmpb (i3 ? mem[at2] : cpu.r[dr])
#define jmpv (i4 ? (i5 ? (word)(jmpb - doff) : (word)(jmpb + doff)) : jmpb)

#define alu_post if (icc) set_cc(r); \
                 tmp2 = cpu.r[dr]; \
                 if (i4) { \
                    tmp = (mem[i3 ? at3 : at2] >> 8) & 0xFF; \
                    if (i5) tmp2 -= tmp; \
                    else tmp2 += tmp; \
                 } \
                 if (imemd) mem[tmp2] = r; \
                 else cpu.r[dr] = r

static void step2(void) {
    word at0 = cpu.pc;
    word at1 = at0 + 1;
    cpu.pc += 2;

    word icc = (mem[at0] >> 7) & 0x1;
    word i2 = (mem[at0] >> 6) & 0x1;
    word imemd = (mem[at0] >> 5) & 0x1;
    word imem1 = (mem[at0] >> 4) & 0x1;
    word imem2 = (mem[at0] >> 3) & 0x1;
    word i3 = (mem[at0] >> 2) & 0x1;
    word i4 = (mem[at0] >> 1) & 0x1;
    word i5 = mem[at0] & 0x1;

    word dr = (mem[at1] >> 12) & 0xF;
    word or1 = (mem[at1] >> 8) & 0xF;
    word or2 = (mem[at1] >> 4) & 0xF;

    word at2 = at1 + 1;
    word at3 = at2 + 1;

    word opc = (mem[at0] >> 8) & 0xFF;

    word offw = i4 ? mem[i3 ? at3 : at2] : 0;
    word soff = offw & 0xFF;
    word doff = (offw >> 8) & 0xFF;

    int64_t timet;

    cpu.ir = mem[at0];
    if ((mem[at1] & 0xF) != 0) bad(at0);  /* second-word low nibble reserved */
    if (i3) {
        /* i3 = third word holds a 16-bit immediate; only some ops take it */
        switch (opc) {
        case I2_ADD: case I2_SUB: case I2_MUL: case I2_AND: case I2_OR:
        case I2_XOR: case I2_SHL: case I2_SHR: case I2_SAR: case I2_DIV:
        case I2_MOD: case I2_CMP: case I2_SDIV: case I2_SMOD:
        case I2_ADC: case I2_SBB:
        case I2_MULH: case I2_MULHS: case I2_DIVMOD:
        case I2_ROL: case I2_ROR: case I2_RCR:
        case I2_BSET: case I2_BCLR: case I2_TST: case I2_BTST:
            if (i2) bad(at0);        /* i2 and i3 are mutually exclusive */
            break;
        case I2_JMP: case I2_JN: case I2_JZ: case I2_JP:
        case I2_JNZ: case I2_JLE: case I2_JGE:
        case I2_UEXT: case I2_JC: case I2_JNC:
        /* target replaces r[dr] */
        case I2_LOAD: case I2_LOAD2: case I2_STOR: case I2_STOR2:
        case I2_IN: case I2_OUT: case I2_PUSH: case I2_MOVE:
        case I2_CALL: case I2_ID:
        case I2_LDB: case I2_STB: case I2_PEEK:
        case I2_PUSHM: case I2_POPM: case I2_JAL:
        /* immediate replaces r[or1] */
            break;
        default:
            bad(at0);
        }
        cpu.pc++;
    }
    if (i5 && !i4) bad(at0);      /* sign without offset word */
    if (i4) {
        /* i4 = extra offset word (low = or1/src off, high = dr/dst off, sign i5).
         * Unused half must be zero so silent ignores trap. */
        switch (opc) {
        case I2_ADD: case I2_SUB: case I2_MUL: case I2_AND: case I2_OR:
        case I2_XOR: case I2_NOT: case I2_NEG:
        case I2_SHL: case I2_SHR: case I2_SAR: case I2_DIV:
        case I2_MOD: case I2_SDIV: case I2_SMOD:
        case I2_ADC: case I2_SBB:
        case I2_MULH: case I2_MULHS:
        case I2_ROL: case I2_ROR: case I2_RCR:
        case I2_BSET: case I2_BCLR:
            if (!imemd && doff != 0) bad(at0);
            break;
        case I2_DIVMOD:
            if (imemd || doff != 0) bad(at0);  /* quot to plain reg only */
            break;
        case I2_CMP: case I2_TST: case I2_BTST:
        case I2_LOAD: case I2_LOAD2: case I2_MOVE: case I2_ID:
        case I2_IN: case I2_PUSH: case I2_CALL:
        case I2_LDB: case I2_PEEK: case I2_JAL:
            if (doff != 0) bad(at0);
            break;
        case I2_STOR: case I2_STOR2: case I2_STB: case I2_OUT: case I2_SWAP:
            break;                  /* both halves used */
        case I2_JMP: case I2_JN: case I2_JZ: case I2_JP:
        case I2_JNZ: case I2_JLE: case I2_JGE: case I2_JC: case I2_JNC:
        case I2_UEXT: case I2_RET:
            if (soff != 0) bad(at0);
            break;
        case I2_POP:
            if (doff != 0) bad(at0);
            break;
        case I2_PUSHM: case I2_POPM:
            if (soff != 0 || doff != 0) bad(at0);
            break;
        default:                    /* NOP, HALT */
            bad(at0);
        }
    }
    /* non-ALU ops must not carry ALU-only mem/i2 flags */
    switch (opc) {
    case I2_ADD: case I2_SUB: case I2_MUL: case I2_AND: case I2_OR:
    case I2_XOR: case I2_NOT: case I2_NEG:
    case I2_SHL: case I2_SHR: case I2_SAR: case I2_DIV:
    case I2_MOD: case I2_CMP: case I2_SDIV: case I2_SMOD:
    case I2_ADC: case I2_SBB:
    case I2_MULH: case I2_MULHS: case I2_DIVMOD:
    case I2_ROL: case I2_ROR: case I2_RCR:
    case I2_BSET: case I2_BCLR: case I2_TST: case I2_BTST:
        break;
    default:
        if (i2 || imemd || imem1 || imem2) bad(at0);
        break;
    }
    /* o2 is unused by NOT/NEG: reject stray flags there */
    if ((opc == I2_NOT || opc == I2_NEG) && (i2 || imem2)) bad(at0);
    /* CMP family has no destination: reject imemd */
    if ((opc == I2_CMP || opc == I2_TST || opc == I2_BTST) && imemd) bad(at0);
    if (i4) cpu.pc++;

    switch (opc) {
    case I2_NOP: {
        break;
    }

    case I2_HALT: {
        running = false;
        break;
    }

    case I2_UEXT: {
        cpu.rs0 = cpu.rs0 & 0xFFFE;
        cpu.pc = jmpv;
        break;
    }

    case I2_ADD: {
        alu_pre;
        r = a + b;
        if (icc) cpu.cc[C] = r < a;
        alu_post;
        break;
    }

    case I2_SUB: {
        alu_pre;
        r = a - b;
        if (icc) cpu.cc[C] = b > a;
        alu_post;
        break;
    }

    case I2_MUL: {
        alu_pre;
        r = (word)((uint32_t)a * b);
        alu_post;
        break;
    }

    case I2_AND: {
        alu_pre;
        r = a & b;
        alu_post;
        break;
    }

    case I2_OR: {
        alu_pre;
        r = a | b;
        alu_post;
        break;
    }

    case I2_NOT: {
        alu_pre;
        r = ~a;
        alu_post;
        break;
    }

    case I2_NEG: {   /* rd = -rs */
        alu_pre;
        r = (word)(0u - a);
        if (icc) cpu.cc[C] = (a != 0);   /* borrow out */
        alu_post;
        break;
    }

    case I2_XOR: {
        alu_pre;
        r = a ^ b;
        alu_post;
        break;
    }

    case I2_JMP: {
        cpu.pc = jmpv;
        break;
    }

    case I2_JN: {
        if (cpu.cc[N]) cpu.pc = jmpv;
        break;
    }

    case I2_JZ: {
        if (cpu.cc[Z]) cpu.pc = jmpv;
        break;
    }

    case I2_JP: {
        if (cpu.cc[P]) cpu.pc = jmpv;
        break;
    }

    case I2_LOAD: {
        cpu.r[dr] = mem[or1v];
        if (icc) set_cc(cpu.r[dr]);
        break;
    }

    case I2_LOAD2: {
        cpu.r[dr] = mem[mem[or1v]];
        if (icc) set_cc(cpu.r[dr]);
        break;
    }

    case I2_STOR: {
        mem[droff] = or1v;
        if (icc) set_cc(or1v);
        break;
    }

    case I2_STOR2: {
        mem[mem[droff]] = or1v;
        if (icc) set_cc(or1v);
        break;
    }

    case I2_IN: {
        switch (or1v) {
        case 0x0:
            fflush(stdout);
            cpu.r[dr] = (word)getchar();
            if (icc) set_cc(cpu.r[dr]);
            break;
        case 0x70:
            cpu.r[dr] = (word)(gettime() & 0xFFFF);
            break;
        case 0x71:
            cpu.r[dr] = (word)((gettime() >> 16) & 0xFFFF);
            break;
        case 0x72:
            cpu.r[dr] = (word)((gettime() >> 32) & 0xFFFF);
            break;
        case 0x73:
            cpu.r[dr] = (word)((gettime() >> 48) & 0xFFFF);
            break;
        default:
            bad(at0);
        }
        break;
    }

    case I2_OUT: {
        switch (droff) {
        case 0x0:
            putchar((char)or1v);
            break;
        case 0x70:
            timet = gettime();
            timet = timet & 0xFFFFFFFFFFFF0000;
            timet = timet + or1v;
            settime(timet);
            break;
        case 0x71:
            timet = gettime();
            timet = timet & 0xFFFFFFFF0000FFFF;
            timet = timet + ((int64_t)or1v << 16);
            settime(timet);
            break;
        case 0x72:
            timet = gettime();
            timet = timet & 0xFFFF0000FFFFFFFF;
            timet = timet + ((int64_t)or1v << 32);
            settime(timet);
            break;
        case 0x73:
            timet = gettime();
            timet = timet & 0x0000FFFFFFFFFFFF;
            timet = timet + ((int64_t)or1v << 48);
            settime(timet);
            break;
        default:
            bad(at0);
        }
        break;
    }

    case I2_PUSH: {
        mem[cpu.r[dr]--] = or1v;
        break;
    }

    case I2_POP: {
        word sp = cpu.r[or1];
        word addr = i4 ? (i5 ? (word)(sp + 1 - soff) : (word)(sp + 1 + soff))
                       : (word)(sp + 1);
        cpu.r[dr] = mem[addr];
        cpu.r[or1] = (word)(sp + 1);
        if (icc) set_cc(cpu.r[dr]);
        break;
    }

    case I2_MOVE: {
        cpu.r[dr] = or1v;
        if (icc) set_cc(cpu.r[dr]);
        break;
    }

    case I2_SWAP: {
        word t = droff;
        cpu.r[dr] = or1v;
        cpu.r[or1] = t;
        if (icc) set_cc(cpu.r[dr]);
        break;
    }

    case I2_CALL: {
        mem[cpu.r[dr]--] = cpu.pc;      /* pc already points past i3/i4 words */
        cpu.pc = or1v;
        break;
    }

    case I2_RET: {
        word sp = (word)(cpu.r[dr] + 1);
        cpu.pc = mem[sp];
        cpu.r[dr] = i4 ? (i5 ? (word)(sp - doff) : (word)(sp + doff)) : sp;
        break;
    }

    case I2_ID: {
        word v;
        switch (or1v) {
        case ID_MAX:      v = ID_NAME2; break;
        case ID_VERSION:  v = BOB16_VERSION; break;
        case ID_FEATURES: v = FEAT_MEMOPS | FEAT_IMM16 | FEAT_SDIV |
                              FEAT_IO | FEAT_STACK | FEAT_OFF |
                              FEAT_EXTALU | FEAT_MEMB | FEAT_STACK2 |
                              FEAT_JX; break;
        case ID_MEMTOP:   v = (word)(MEM_WORDS - 1); break;
        case ID_RS0:      v = cpu.rs0; break;
        case ID_REGS:     v = (8 << 8) | 16; break;
        case ID_OPCODES:  v = I2_JAL + 1; break;
        case ID_PORTS:    v = PORTF_IO | PORTF_TIME; break;
        case ID_NAME0:    v = ('B' << 8) | 'O'; break;
        case ID_NAME1:    v = ('B' << 8) | '-'; break;
        case ID_NAME2:    v = ('1' << 8) | '6'; break;
        default:          v = 0;
        }
        cpu.r[dr] = v;
        if (icc) set_cc(v);
        break;
    }

    case I2_SHL: {
        alu_pre;
        r = b > 15 ? 0 : (word)(a << b);
        alu_post;
        break;
    }

    case I2_SHR: {   /* logical */
        alu_pre;
        r = b > 15 ? 0 : (word)(a >> b);
        alu_post;
        break;
    }

    case I2_SAR: {   /* arithmetic: sign bit fills in */
        alu_pre;
        if (b > 15) b = 15;
        r = (word)((int16_t)a >> b);
        alu_post;
        break;
    }

    case I2_DIV: {   /* unsigned */
        alu_pre;
        if (b == 0) {
            bad(at0);
        }
        r = a / b;
        alu_post;
        break;
    }

    case I2_MOD: {   /* unsigned */
        alu_pre;
        if (b == 0) {
            bad(at0);
        }
        r = a % b;
        alu_post;
        break;
    }

    case I2_CMP: {   /* compares or1 with b, writes only the flags */
        alu_src;
        cpu.cc[N] = (int16_t)a < (int16_t)b;
        cpu.cc[Z] = a == b;
        cpu.cc[P] = (int16_t)a > (int16_t)b;
        break;
    }

    case I2_TST: {   /* and without write, flags only */
        alu_src;
        {
            word t = (word)(a & b);
            cpu.cc[N] = ((int16_t)t) < 0;
            cpu.cc[Z] = t == 0;
            cpu.cc[P] = ((int16_t)t) > 0;
        }
        break;
    }

    case I2_BTST: {   /* C = tested bit; N/Z/P from masked result */
        alu_src;
        {
            unsigned bit = b & 15;
            word t = (word)(a & ((word)1u << bit));
            cpu.cc[C] = (a >> bit) & 1u;
            cpu.cc[N] = ((int16_t)t) < 0;
            cpu.cc[Z] = t == 0;
            cpu.cc[P] = ((int16_t)t) > 0;
        }
        break;
    }

    case I2_JNZ: {
        if (cpu.cc[N] || cpu.cc[P]) cpu.pc = jmpv;
        break;
    }

    case I2_JLE: {   /* N or Z */
        if (cpu.cc[N] || cpu.cc[Z]) cpu.pc = jmpv;
        break;
    }

    case I2_JGE: {   /* Z or P */
        if (cpu.cc[Z] || cpu.cc[P]) cpu.pc = jmpv;
        break;
    }

    case I2_SDIV: {   /* signed, truncates toward zero */
        alu_pre;
        if (b == 0) {
            bad(at0);
        }
        r = (word)((int16_t)a / (int16_t)b);
        alu_post;
        break;
    }

    case I2_SMOD: {   /* remainder takes the sign of the dividend */
        alu_pre;
        if (b == 0) {
            bad(at0);
        }
        r = (word)((int16_t)a % (int16_t)b);
        alu_post;
        break;
    }

    case I2_ADC: {
        alu_pre;
        uint32_t t = (uint32_t)a + b + cpu.cc[C];   /* a + b + carry-in */
        r = (word)t;
        if (icc) cpu.cc[C] = t > 0xFFFF;            /* carry-out */
        alu_post;
        break;
    }

    case I2_SBB: {
        alu_pre;
        uint32_t bw = (uint32_t)b + cpu.cc[C];      /* b + borrow-in */
        r = (word)(a - bw);
        if (icc) cpu.cc[C] = bw > a;                /* borrow-out */
        alu_post;
        break;
    }

    case I2_JC: {
        if (cpu.cc[C]) cpu.pc = jmpv;
        break;
    }

    case I2_JNC: {
        if (!cpu.cc[C]) cpu.pc = jmpv;
        break;
    }

    case I2_MULH: {   /* upper 16 of unsigned product */
        alu_pre;
        r = (word)(((uint32_t)a * b) >> 16);
        alu_post;
        break;
    }

    case I2_MULHS: {   /* upper 16 of signed product */
        alu_pre;
        r = (word)(((int32_t)(int16_t)a * (int16_t)b) >> 16);
        alu_post;
        break;
    }

    case I2_DIVMOD: {   /* quot -> rd, rem -> or1 (in-out) */
        alu_pre;
        if (b == 0) {
            bad(at0);
        }
        {
            word abase = cpu.r[or1];
            if (i4) {
                if (i5) abase -= soff;
                else abase += soff;
            }
            word m = (word)(a % b);
            r = (word)(a / b);
            if (imem1) mem[abase] = m;
            else cpu.r[or1] = m;
        }
        if (icc) set_cc(r);
        cpu.r[dr] = r;   /* validated: plain reg dest only */
        break;
    }

    case I2_ROL: {
        alu_pre;
        {
            unsigned n = b & 15;
            r = n ? (word)((a << n) | (a >> (16 - n))) : a;
        }
        alu_post;
        break;
    }

    case I2_ROR: {
        alu_pre;
        {
            unsigned n = b & 15;
            r = n ? (word)((a >> n) | (a << (16 - n))) : a;
        }
        alu_post;
        break;
    }

    case I2_RCR: {   /* rotate right through carry; C in, C out iff icc */
        alu_pre;
        {
            unsigned n = b % 17;
            word cout = cpu.cc[C];
            if (n != 0) {
                uint32_t v = ((uint32_t)cpu.cc[C] << 16) | a;
                v = ((v >> n) | (v << (17 - n))) & 0x1FFFFu;
                r = (word)(v & 0xFFFFu);
                cout = (word)((v >> 16) & 1u);
            } else {
                r = a;
            }
            if (icc) cpu.cc[C] = cout;
        }
        alu_post;
        break;
    }

    case I2_BSET: {
        alu_pre;
        r = (word)(a | ((word)1u << (b & 15)));
        alu_post;
        break;
    }

    case I2_BCLR: {
        alu_pre;
        r = (word)(a & ~((word)1u << (b & 15)));
        alu_post;
        break;
    }

    case I2_LDB: {   /* zero-extending byte load; word = addr>>1 */
        word addr = or1v;
        word wd = mem[addr >> 1];
        cpu.r[dr] = (addr & 1) ? ((wd >> 8) & 0xFF) : (wd & 0xFF);
        if (icc) set_cc(cpu.r[dr]);
        break;
    }

    case I2_STB: {   /* byte store, preserves the other half */
        word addr = droff;
        word v = or1v & 0xFF;
        word wd = mem[addr >> 1];
        mem[addr >> 1] = (addr & 1) ? (word)((wd & 0xFF) | (v << 8))
                                    : (word)((wd & 0xFF00) | v);
        if (icc) set_cc(or1v);
        break;
    }

    case I2_PEEK: {   /* non-destructive top-of-stack read */
        cpu.r[dr] = mem[(word)(or1v + 1)];
        if (icc) set_cc(cpu.r[dr]);
        break;
    }

    case I2_PUSHM: {   /* push every reg in mask, ascending */
        word mask = or1v;
        for (int n = 0; n < 16; n++) {
            if ((mask >> n) & 1u) mem[cpu.r[dr]--] = cpu.r[n];
        }
        break;
    }

    case I2_POPM: {   /* pop every reg in mask, descending */
        word mask = or1v;
        for (int n = 15; n >= 0; n--) {
            if ((mask >> n) & 1u) cpu.r[n] = mem[++cpu.r[dr]];
        }
        break;
    }

    case I2_JAL: {   /* link in register (vs CALL to stack) */
        word ret = cpu.pc;   /* past i3/i4 words */
        cpu.pc = or1v;
        cpu.r[dr] = ret;
        break;
    }

    default:
        bad(at0);
    }
}

static void step(void) {
    if (cpu.rs0 & 0x1) {
        step2();
    } else {
        step1();
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s program.bin\n", argv[0]);
        return 1;
    }

    load_program(argv[1]);

    cpu.rs0 = 0;
    int64_t t = (int64_t)time(NULL);
    virtual_time = t;
    actual_time = t;
    while (running) {
        step();
    }

    fflush(stdout);
    return 0;
}
