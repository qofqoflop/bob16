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
enum { N, Z, P };
enum { I2_NOP, I2_HALT, I2_UEXT, I2_ADD,
       I2_SUB, I2_MUL, I2_AND, I2_OR,
       I2_NOT, I2_XOR, I2_JMP, I2_JN,
       I2_JZ, I2_JP, I2_LOAD, I2_LOAD2,
       I2_STOR, I2_STOR2, I2_IN, I2_OUT,
       I2_PUSH, I2_POP, I2_MOVE, I2_SWAP,
       I2_CALL, I2_RET, I2_ID, I2_SHL,
       I2_SHR, I2_SAR, I2_DIV, I2_MOD,
       I2_CMP, I2_JNZ, I2_JLE, I2_JGE,
       I2_SDIV, I2_SMOD };

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
       FEAT_STACK  = 1 << 4 }; /* push / pop / call / ret */

enum { PORTF_IO   = 1 << 0,
       PORTF_TIME = 1 << 1 };

#define BOB16_VERSION 0x0100   /* 1.0 */

static struct {
    word r[16];
    word ir;
    word pc;
    word rs0;
    bool cc[3];
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

#define alu_src word a, b; \
                a = cpu.r[or1]; \
                b = i2 ? or2 : i3 ? mem[at2] : cpu.r[or2]; \
                if (imem1) a = mem[a]; \
                if (imem2) b = mem[b]

#define alu_pre word r; alu_src

/* With i3 set, the third word replaces the or1 operand of non-ALU
 * instructions (move/load/stor/in/out/push/call/id) */
#define or1v (i3 ? mem[at2] : cpu.r[or1])

#define alu_post if (icc) set_cc(r); \
                 if (imemd) mem[cpu.r[dr]] = r; \
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

    word dr = (mem[at1] >> 12) & 0xF;
    word or1 = (mem[at1] >> 8) & 0xF;
    word or2 = (mem[at1] >> 4) & 0xF;

    word at2 = at1 + 1;
    word opc = (mem[at0] >> 8) & 0xFF;

    int64_t timet;

    cpu.ir = mem[at0];
    if (i3) {
        /* i3 = third word holds a 16-bit immediate; only some ops take it */
        switch (opc) {
        case I2_ADD: case I2_SUB: case I2_MUL: case I2_AND: case I2_OR:
        case I2_XOR: case I2_SHL: case I2_SHR: case I2_SAR: case I2_DIV:
        case I2_MOD: case I2_CMP: case I2_SDIV: case I2_SMOD:
            if (i2) bad(at0);        /* i2 and i3 are mutually exclusive */
            break;
        case I2_JMP: case I2_JN: case I2_JZ: case I2_JP:
        case I2_JNZ: case I2_JLE: case I2_JGE:
        case I2_UEXT:                    /* target replaces r[dr] */
        case I2_LOAD: case I2_LOAD2: case I2_STOR: case I2_STOR2:
        case I2_IN: case I2_OUT: case I2_PUSH: case I2_MOVE:
        case I2_CALL: case I2_ID:        /* immediate replaces r[or1] */
            break;
        default:
            bad(at0);
        }
        cpu.pc++;
    }

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
        cpu.pc = i3 ? mem[at2] : cpu.r[dr];
        break;
    }

    case I2_ADD: {
        alu_pre;
        r = a + b;
        alu_post;
        break;
    }

    case I2_SUB: {
        alu_pre;
        r = a - b;
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

    case I2_XOR: {
        alu_pre;
        r = a ^ b;
        alu_post;
        break;
    }

    case I2_JMP: {
        cpu.pc = i3 ? mem[at2] : cpu.r[dr];
        break;
    }

    case I2_JN: {
        if (cpu.cc[N]) cpu.pc = i3 ? mem[at2] : cpu.r[dr];
        break;
    }

    case I2_JZ: {
        if (cpu.cc[Z]) cpu.pc = i3 ? mem[at2] : cpu.r[dr];
        break;
    }

    case I2_JP: {
        if (cpu.cc[P]) cpu.pc = i3 ? mem[at2] : cpu.r[dr];
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
        mem[cpu.r[dr]] = or1v;
        if (icc) set_cc(or1v);
        break;
    }

    case I2_STOR2: {
        mem[mem[cpu.r[dr]]] = or1v;
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
        switch (cpu.r[dr]) {
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
        cpu.r[dr] = mem[++cpu.r[or1]];
        if (icc) set_cc(cpu.r[dr]);
        break;
    }

    case I2_MOVE: {
        cpu.r[dr] = or1v;
        if (icc) set_cc(cpu.r[dr]);
        break;
    }

    case I2_SWAP: {
        word t = cpu.r[dr];
        cpu.r[dr] = cpu.r[or1];
        cpu.r[or1] = t;
        if (icc) set_cc(cpu.r[dr]);
        break;
    }

    case I2_CALL: {
        mem[cpu.r[dr]--] = cpu.pc;      /* pc already points past the i3 word */
        cpu.pc = or1v;
        break;
    }

    case I2_RET: {
        cpu.pc = mem[++cpu.r[dr]];
        break;
    }

    case I2_ID: {
        word v;
        switch (or1v) {
        case ID_MAX:      v = ID_NAME2; break;
        case ID_VERSION:  v = BOB16_VERSION; break;
        case ID_FEATURES: v = FEAT_MEMOPS | FEAT_IMM16 | FEAT_SDIV |
                              FEAT_IO | FEAT_STACK; break;
        case ID_MEMTOP:   v = (word)(MEM_WORDS - 1); break;
        case ID_RS0:      v = cpu.rs0; break;
        case ID_REGS:     v = (8 << 8) | 16; break;
        case ID_OPCODES:  v = I2_SMOD + 1; break;
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

    case I2_JNZ: {
        if (cpu.cc[N] || cpu.cc[P]) cpu.pc = i3 ? mem[at2] : cpu.r[dr];
        break;
    }

    case I2_JLE: {   /* N or Z */
        if (cpu.cc[N] || cpu.cc[Z]) cpu.pc = i3 ? mem[at2] : cpu.r[dr];
        break;
    }

    case I2_JGE: {   /* Z or P */
        if (cpu.cc[Z] || cpu.cc[P]) cpu.pc = i3 ? mem[at2] : cpu.r[dr];
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
