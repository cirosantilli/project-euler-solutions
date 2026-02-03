import ProjectEulerSolutions.P89

open ProjectEulerSolutions.P89

def main : IO Unit := do
  let text ← IO.FS.readFile "0089_roman.txt"
  let lines := text.splitOn "\n" |>.filter (fun ln => ln != "")
  IO.println (solve lines)
