import ProjectEulerSolutions.P67

open ProjectEulerSolutions.P67

def main : IO Unit := do
  let text ← IO.FS.readFile "0067_triangle.txt"
  IO.println (solve (parseTriangle text))
