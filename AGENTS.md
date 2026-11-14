# AGENTS.md

## Python style

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

## README

The README is not a dump of shit. It is meant to contain information which humans find interesting and useful. In particular, the start of sections must be nice. You have a tendency to create humongously super long useless paragraphs that go in depth into something human readers won't care about. Please don't do that. Use a separate adoc for your notes is that is the case. Do update README as needed, but it has to remain meaningful to human readers.

Also, the README is not an AGENTS.md. You have the creepy tendency of writing AGENT guidelines on the README. If you write something there, it cannot read as agent guidelines.

## Solution explanations

Do not add proof status, proof coverage, or progress reports to solution `.md` files. Keep mathematical and algorithmic explanations and links to source files there.
