import ProjectEulerSolutions.P96

def main : IO Unit := do
  let text ← IO.FS.readFile "0096_sudoku.txt"
  match ProjectEulerSolutions.P96.solve text with
  | some total => IO.println total
  | none => throw (IO.userError "Unsolvable puzzle encountered")
