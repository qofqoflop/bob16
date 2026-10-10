; timestamp.asm - print the current unix timestamp (decimal) and halt.
;
; Uses the 64-bit time ports (0x70 low .. 0x73 high). The value fits in
; 32 bits until 2106, so only the low two words are converted, with a
; base-256 long division by 10 (each step stays in 16 bits).
; Build:  python3 bob16asm.py timestamp.asm
; Run:    ./bob16 timestamp.bin

enter2 main
halt
main:
.mode 2
clr r0                  ; r0 = 0 = console port for `out`
move r10, 0x1000        ; stack
in r1, 0x71             ; hi word of time
in r2, 0x70             ; lo word of time
move r3, 10             ; divisor
clr r4                  ; digit count
or.cc r5, r1, r2
jz print_zero           ; time == 0 -> print '0'

loop:                   ; 32-bit value in r1:r2 -> push one decimal digit
shr r12, r1, 8          ; split into 4 base-256 digits, big end first
and r13, r1, 0xFF
shr r14, r2, 8
and r15, r2, 0xFF
clr r5                  ; running remainder
shl r5, 8
add r5, r12
divmod r11, r5, r3
shl r7, r11, 8
shl r5, 8
add r5, r13
divmod r11, r5, r3
or r7, r11              ; r7 = new hi word
shl r5, 8
add r5, r14
divmod r11, r5, r3
shl r8, r11, 8
shl r5, 8
add r5, r15
divmod r11, r5, r3
or r8, r11              ; r8 = new lo word
move r1, r7
move r2, r8
push r10, r5            ; r5 = remainder = next digit
inc r4
or.cc r11, r1, r2
jnz loop                ; until the value is 0

print:                  ; pop digits (reversed) and print them
cmp r4, 0
jz done
pop r5, r10
add r5, 48              ; to ASCII
out r0, r5
dec r4
jmp print
print_zero:
move r5, 48
out r0, r5
done:
move r5, 10             ; newline
out r0, r5
halt
