# Solution summary

This solution treats the pixel update rule as a linear operator on a toroidal grid.

## Main ideas

1. **Work modulo 7 from the start**
   The final image is only needed modulo 7, so every intermediate value can also be reduced modulo 7.

2. **Use the torus shift operators**
   Let `U`, `D`, `L`, and `R` be the four wraparound shifts. One step is:

   `T = U + D + L + R`

3. **Exploit characteristic 7**
   Over modulo 7 arithmetic, the Frobenius identity gives:

   `T^(7^k) = U^(7^k) + D^(7^k) + L^(7^k) + R^(7^k)`

   because all mixed binomial and multinomial coefficients are divisible by 7.

4. **Decompose `10^12` in base 7**
   If

   `10^12 = sum(d_k * 7^k)`

   then

   `T^(10^12) = product(T^(7^k))^(d_k)`.

   Since each base-7 digit `d_k` is at most 6, the huge exponent becomes only a small number of effective passes.

5. **Apply only the needed jumps**
   For each base-7 digit, the code applies the update with vertical and horizontal jumps of `7^k` (taken modulo the image dimensions) exactly `d_k` times.

## Implementation notes

- `main.py` contains a small **PNG decoder** built only from the Python standard library.
- The solver supports the provided non-interlaced 8-bit grayscale/RGB PNG input.
- The revealed modulo-7 image is also written to `revealed.pgm` so it can be inspected directly.
- The sample wraparound-neighbour relationships from the statement are checked with `assert`.

## Usage

Place the puzzle image at `image.png` next to `main.py`, or pass its path explicitly:

```bash
python3 main.py image.png
