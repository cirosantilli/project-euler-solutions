import ProjectEulerSolutions.P98

open ProjectEulerSolutions.P98

def main : IO Unit := do
  let text ← IO.FS.readFile "0098_words.txt"
  IO.println (solve (parseWords text))
