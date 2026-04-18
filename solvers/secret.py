#!/usr/bin/env bash

'''
wget -O data/project-euler-statements/data/images/bonus_secret_statement.png secret.png https://projecteuler.net/resources/images/bonus_secret_statement.png?1738588439
pypy3 secret.py
'''

import hashlib
import struct
import zlib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MOD7_TABLE = [value % 7 for value in range(25)]
KNOWN_RESULT_HASH = "d0823f806ffc87d57a5a4829199b7a2873ff8bf3d473f5535a3c555f09eef32c"
DISPLAY_LEVELS = bytes([0, 42, 84, 126, 168, 210, 252])


def paeth_predictor(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c



def read_png_mod7(path):
    with open(path, "rb") as handle:
        data = handle.read()

    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG file")

    pos = len(PNG_SIGNATURE)
    width = height = None
    bit_depth = color_type = interlace = None
    idat_parts = []

    while pos < len(data):
        if pos + 8 > len(data):
            raise ValueError("truncated PNG chunk header")
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        pos += 8
        chunk_data = data[pos : pos + length]
        pos += length
        pos += 4  # Skip CRC.

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if compression != 0 or filter_method != 0:
                raise ValueError("unsupported PNG compression or filter method")
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None:
        raise ValueError("missing IHDR chunk")
    if bit_depth != 8:
        raise ValueError("only 8-bit PNG images are supported")
    if interlace != 0:
        raise ValueError("interlaced PNG images are not supported")
    if color_type not in (0, 2):
        raise ValueError("only grayscale and RGB PNG images are supported")

    bytes_per_pixel = 1 if color_type == 0 else 3
    stride = width * bytes_per_pixel
    raw = zlib.decompress(b"".join(idat_parts))
    expected_size = height * (stride + 1)
    if len(raw) != expected_size:
        raise ValueError("unexpected decompressed PNG size")

    rows = []
    previous = bytearray(stride)
    pos = 0
    for _ in range(height):
        filter_type = raw[pos]
        pos += 1
        scanline = raw[pos : pos + stride]
        pos += stride

        recon = bytearray(stride)
        if filter_type == 0:
            recon[:] = scanline
        elif filter_type == 1:
            for i in range(stride):
                left = recon[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                recon[i] = (scanline[i] + left) & 255
        elif filter_type == 2:
            for i in range(stride):
                recon[i] = (scanline[i] + previous[i]) & 255
        elif filter_type == 3:
            for i in range(stride):
                left = recon[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                up = previous[i]
                recon[i] = (scanline[i] + ((left + up) >> 1)) & 255
        elif filter_type == 4:
            for i in range(stride):
                left = recon[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                up = previous[i]
                up_left = previous[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                recon[i] = (scanline[i] + paeth_predictor(left, up, up_left)) & 255
        else:
            raise ValueError("unsupported PNG filter type")

        row = bytearray(width)
        if color_type == 0:
            for x in range(width):
                row[x] = recon[x] % 7
        else:
            j = 0
            for x in range(width):
                r = recon[j]
                g = recon[j + 1]
                b = recon[j + 2]
                j += 3
                if r == g == b:
                    gray = r
                else:
                    gray = (r + g + b) // 3
                row[x] = gray % 7
        rows.append(row)
        previous = recon

    return rows



def neighbors(y, x, height, width):
    return (
        ((y - 1) % height, x),
        ((y + 1) % height, x),
        (y, (x - 1) % width),
        (y, (x + 1) % width),
    )



def step_once(grid, vshift, hshift):
    height = len(grid)
    width = len(grid[0])
    left_index = [(x - hshift) % width for x in range(width)]
    right_index = [(x + hshift) % width for x in range(width)]
    out = [bytearray(width) for _ in range(height)]

    for y in range(height):
        up = grid[(y - vshift) % height]
        down = grid[(y + vshift) % height]
        cur = grid[y]
        out_row = out[y]
        for x in range(width):
            out_row[x] = MOD7_TABLE[up[x] + down[x] + cur[left_index[x]] + cur[right_index[x]]]
    return out



def apply_many_steps_mod7(grid, steps):
    height = len(grid)
    width = len(grid[0])
    result = [bytearray(row) for row in grid]
    power = 1
    remaining = steps

    while remaining:
        digit = remaining % 7
        vshift = power % height
        hshift = power % width
        for _ in range(digit):
            result = step_once(result, vshift, hshift)
        remaining //= 7
        power *= 7
    return result



def write_pgm(path, grid):
    height = len(grid)
    width = len(grid[0])
    with open(path, "wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        for row in grid:
            handle.write(bytes(DISPLAY_LEVELS[value] for value in row))



def grid_digest(grid):
    digest = hashlib.sha256()
    for row in grid:
        digest.update(row)
    return digest.hexdigest()



def run_self_tests():
    labels = {
        (1, 0): "B",
        (2, 0): "A",
        (3, 0): "D",
        (2, 1): "E",
        (2, 9): "C",
        (0, 9): "d",
        (4, 9): "b",
        (5, 9): "a",
        (5, 8): "c",
        (5, 0): "e",
    }

    a_neighbours = {labels[pos] for pos in neighbors(2, 0, 6, 10)}
    small_a_neighbours = {labels[pos] for pos in neighbors(5, 9, 6, 10)}
    assert a_neighbours == {"B", "C", "D", "E"}
    assert small_a_neighbours == {"b", "c", "d", "e"}

    tiny = [
        bytearray([0, 1, 2, 3]),
        bytearray([4, 5, 6, 0]),
        bytearray([1, 2, 3, 4]),
    ]
    slow = [bytearray(row) for row in tiny]
    for _ in range(20):
        slow = step_once(slow, 1, 1)
    fast = apply_many_steps_mod7(tiny, 20)
    assert fast == slow



def main():
    run_self_tests()
    grid = read_png_mod7('../images/bonus_secret_statement.png')
    revealed = apply_many_steps_mod7(grid, 10 ** 12)
    write_pgm("secret.pgm", revealed)


if __name__ == "__main__":
    main()
