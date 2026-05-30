import ProjectEulerSolutions.P67

open ProjectEulerSolutions.P67

def readTriangle (text : String) : IO ProjectEulerStatements.P18.Triangle := do
  match toTriangle? (parseTriangle text) with
  | some triangle => pure triangle
  | none => throw (IO.userError "invalid triangle input")

def main : IO Unit := do
  let text ← IO.FS.readFile "0067_triangle.txt"
  let triangle ← readTriangle text
  IO.println (solve triangle)
