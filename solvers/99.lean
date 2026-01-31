import ProjectEulerSolutions.P99

open ProjectEulerSolutions.P99

def main : IO Unit := do
  let text ← IO.FS.readFile "0099_base_exp.txt"
  IO.println (solve (parsePairs text))
