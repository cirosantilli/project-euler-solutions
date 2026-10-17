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
