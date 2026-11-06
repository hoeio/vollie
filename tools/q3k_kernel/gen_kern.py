# Generates the inline-asm bodies for arch/arm/quants.c (Vollie r10):
#   python3 gen_kern.py          -> ggml_vec_dot_q3_K_q8_K_x4_kern (GGML_Q3K_KERN=1), the shipped kernel
#   python3 gen_kern.py scalar   -> the GGML_CPU_BENCH "scalar-assisted" prototype (bench only)
# One Q3_K block x 4 rows per loop iteration; registers fixed (see the C comment above the asm).
# Value byte per plane k of half hf: (q2 + 4h) << p, p = 0, 2, 4, 0 for k = 0..3; the group sum is seeded with
# -(4 << p) * bsum[g] and its int32 scale is pre-shifted by 4 - p, so isum = 16 x the block sum (exact).
# scalar: planes 0 and 1 of the NEXT block are built by the integer units (64-bit SWAR on two rows' words, the same
# bytes) into a 512-byte L1 buffer (%[bn]) while NEON runs this block from %[bc]; the two streams are interleaved.
import sys
R = ['%[x0]', '%[x1]', '%[x2]', '%[x3]']
P = [0, 2, 4, 0]

def tr4(a, i, t, o):  # in regs i[0..3], temps t[0..3], out o[0..3] (out[w] lane r = word w of row r)
    a(f'trn1 v{t[0]}.4s, v{i[0]}.4s, v{i[1]}.4s'); a(f'trn2 v{t[1]}.4s, v{i[0]}.4s, v{i[1]}.4s')
    a(f'trn1 v{t[2]}.4s, v{i[2]}.4s, v{i[3]}.4s'); a(f'trn2 v{t[3]}.4s, v{i[2]}.4s, v{i[3]}.4s')
    a(f'trn1 v{o[0]}.2d, v{t[0]}.2d, v{t[2]}.2d'); a(f'trn1 v{o[1]}.2d, v{t[1]}.2d, v{t[3]}.2d')
    a(f'trn2 v{o[2]}.2d, v{t[0]}.2d, v{t[2]}.2d'); a(f'trn2 v{o[3]}.2d, v{t[1]}.2d, v{t[3]}.2d')

def unpack(hf, k, t, h, q):  # ops on temp t from H = v{h}, Q = v{q}
    if hf == 0:
        if k == 0: return [f'shl v{t}.16b, v{h}.16b, #2', f'and v{t}.16b, v{t}.16b, v29.16b', f'bit v{t}.16b, v{q}.16b, v26.16b']
        if k == 1: return [f'shl v{t}.16b, v{h}.16b, #3', f'and v{t}.16b, v{t}.16b, v30.16b', f'bit v{t}.16b, v{q}.16b, v27.16b']
        if k == 2: return [f'shl v{t}.16b, v{h}.16b, #4', f'and v{t}.16b, v{t}.16b, v31.16b', f'bit v{t}.16b, v{q}.16b, v28.16b']
        return [f'ushr v{t}.16b, v{h}.16b, #1', f'and v{t}.16b, v{t}.16b, v29.16b', f'sri v{t}.16b, v{q}.16b, #6']
    if k == 0: return [f'ushr v{t}.16b, v{h}.16b, #2', f'and v{t}.16b, v{t}.16b, v29.16b', f'bit v{t}.16b, v{q}.16b, v26.16b']
    if k == 1: return [f'ushr v{t}.16b, v{h}.16b, #1', f'and v{t}.16b, v{t}.16b, v30.16b', f'bit v{t}.16b, v{q}.16b, v27.16b']
    if k == 2: return [f'and v{t}.16b, v{h}.16b, v31.16b', f'bit v{t}.16b, v{q}.16b, v28.16b']
    return [f'ushr v{t}.16b, v{h}.16b, #5', f'sri v{t}.16b, v{q}.16b, #6']

def neon_block(scalar):  # one block; scalar = planes 0, 1 come from the buffer %[bc]
    L = []; a = L.append
    for r in range(4): a(f'ldur q{r}, [{R[r]}, #94]')   # scales[12] + d (16 bytes from 94, inside the block)
    for r in range(4): a(f'ext v{r}.16b, v{r}.16b, v{r}.16b, #2')
    tr4(a, [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3])     # v0..v2 = aux0..2 of rows 0-3, v3 = d | 2 qs bytes
    a('ldr s17, [%[y]]')
    a('xtn v3.4h, v3.4s'); a('fcvtl v22.4s, v3.4h'); a('fmul v22.4s, v22.4s, v17.s[0]')   # D = fp16(x.d) * y.d
    # utmp[m] of the 4 rows (6-bit scales, minus 32 later): U = v12..v15
    a('and v12.16b, v0.16b, v25.16b'); a('shl v4.16b, v2.16b, #4'); a('bit v12.16b, v4.16b, v28.16b')
    a('and v13.16b, v1.16b, v25.16b'); a('shl v5.16b, v2.16b, #2'); a('bit v13.16b, v5.16b, v28.16b')
    a('ushr v14.16b, v0.16b, #4'); a('bit v14.16b, v2.16b, v28.16b')
    a('ushr v15.16b, v1.16b, #4'); a('ushr v6.16b, v2.16b, #2'); a('bit v15.16b, v6.16b, v28.16b')
    first = True
    for c in range(2):
        for r in range(4): a(f'ldr q{4 + r}, [{R[r]}, #{16*c}]')
        tr4(a, [4, 5, 6, 7], [8, 9, 10, 11], [0, 1, 2, 3])     # H (hmask bytes 16c..16c+15), lane = row
        for hf in range(2):
            for r in range(4): a(f'ldr q{4 + r}, [{R[r]}, #{32 + 32*hf + 16*c}]')
            tr4(a, [4, 5, 6, 7], [8, 9, 10, 11], [4, 5, 6, 7])
            # scales of groups g = 8hf + 2k + c, k = 0..3, rows 0..3 -> int32 x (16 >> p)
            a(f'ldr q19, [%[ix], #{16*c}]')
            a(f'tbl v8.16b, {{ v{12 + 2*hf}.16b, v{13 + 2*hf}.16b }}, v19.16b')
            a('ssubl v9.8h, v8.8b, v24.8b'); a('ssubl2 v10.8h, v8.16b, v24.16b')
            a('sshll v8.4s, v9.4h, #4'); a('sshll2 v9.4s, v9.8h, #2'); a('sshll2 v11.4s, v10.8h, #4'); a('sshll v10.4s, v10.4h, #0')
            for k in range(4):
                g = 8*hf + 2*k + c
                p = P[k]
                a(f'ldr q17, [%[y], #{4 + 16*g}]')
                a(f'ldrsh w9, [%[y], #{260 + 2*g}]'); a(f'neg w9, w9, lsl #{2 + p}'); a('dup v18.4s, w9')
                for w in range(4):
                    t = 19 + w % 3
                    if scalar and k < 2:
                        a(f'ldr q{t}, [%[bc], #{(4*c + 2*hf + k)*64 + 16*w}]')
                    else:
                        for op in unpack(hf, k, t, w, 4 + w): a(op)
                    a(f'sdot v18.4s, v{t}.16b, v17.4b[{w}]')
                if first: a(f'mul v16.4s, v18.4s, v{8 + k}.4s'); first = False
                else: a(f'mla v16.4s, v18.4s, v{8 + k}.4s')
    a('sshr v16.4s, v16.4s, #4'); a('scvtf v16.4s, v16.4s'); a('fmla v23.4s, v16.4s, v22.4s')
    return L

def scalar_stream(off):  # planes 0, 1 of the block at x_r + off into %[bn] (same layout as the NEON vectors)
    L = []; a = L.append
    for c in range(2):
        for w in range(4):
            for rp in range(2):
                ra, rb = R[2*rp], R[2*rp + 1]
                a(f'ldur w10, [{ra}, #{off + 16*c + 4*w}]'); a(f'ldur w11, [{rb}, #{off + 16*c + 4*w}]')
                a('orr x10, x10, x11, lsl #32')                       # hmask word w of rows 2rp, 2rp+1
                for hf in range(2):
                    o = off + 32 + 32*hf + 16*c + 4*w
                    a(f'ldur w12, [{ra}, #{o}]'); a(f'ldur w13, [{rb}, #{o}]')
                    a('orr x12, x12, x13, lsl #32')                   # qs word
                    for k in range(2):
                        b = 4*hf + k
                        a(f'and x14, x12, #{"0x0303030303030303" if k == 0 else "0x0c0c0c0c0c0c0c0c"}')
                        a(f'and x15, x16, x10, lsr #{b}')             # x16 = 0x0101..01: the hmask bit
                        a(f'orr x14, x14, x15, lsl #{2*k + 2}')
                        a(f'str x14, [%[bn], #{(4*c + 2*hf + k)*64 + 16*w + 8*rp}]')
    return L

def merge(n, s):  # interleave s into n evenly
    out, j = [], 0
    for i, l in enumerate(n):
        out.append(l)
        while j < len(s) and j * len(n) < (i + 1) * len(s):
            out.append(s[j]); j += 1
    return out + s[j:]

def tail(L):
    for r in range(4): L.append(f'add {R[r]}, {R[r]}, #110')
    L += ['add %[y], %[y], #292', 'subs %w[nb], %w[nb], #1', 'b.ne 1b']
    return L

if len(sys.argv) > 1 and sys.argv[1] == 'scalar':
    L = ['mov x16, #0x0101010101010101'] + scalar_stream(0) + ['mov x17, %[bc]', 'mov %[bc], %[bn]', 'mov %[bn], x17', '1:']
    L += merge(neon_block(True), scalar_stream(110))
    L += ['mov x17, %[bc]', 'mov %[bc], %[bn]', 'mov %[bn], x17']
    L = tail(L)
else:
    L = tail(['1:'] + neon_block(False))
print('\n'.join(f'        "{l}\\n"' for l in L))
