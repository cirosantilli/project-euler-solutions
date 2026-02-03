import ProjectEulerSolutions.P42

open ProjectEulerSolutions.P42

def main : IO Unit := do
  let text ← IO.FS.readFile "0042_words.txt"
  IO.println (solve (parseWords text))
