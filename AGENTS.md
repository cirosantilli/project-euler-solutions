# Python style

For multiline strings, start the first line on an empty line, good:

```
def naive(n: int) -> int:
    """
    @formalsw: Requires(n >= 0)
    @formalsw: Returns
    """
```

bad:

```
def naive(n: int) -> int:
    """@formalsw: Requires(n >= 0)
    @formalsw: Returns
    """
```

# Specification comments

Keep `spec.yaml` comment fields limited to current design considerations, such as
possible argument generalizations or constraints the formal specification cannot
express. Do not put Python signatures, change history, or progress reports there.

# Reference docstrings

Do not repeat `spec.yaml` in reference `naive()` docstrings. Keep formal
annotations and document only Python-specific considerations there.

# Solution explanations

Do not add proof status, proof coverage, or progress reports to solution `.md` files. Keep mathematical and algorithmic explanations and links to source files there.
