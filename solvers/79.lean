import ProjectEulerSolutions.P79

open ProjectEulerSolutions.P79

def main : IO Unit := do
  let text ← IO.FS.readFile "0079_keylog.txt"
  let attempts := text.splitOn "\n" |>.filter (fun ln => ln != "")
  IO.println (solve attempts)
