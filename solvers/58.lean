import ProjectEulerSolutions.P58
import ProjectEulerSolutions.Lib.SpiralRowCertificates.All

open ProjectEulerSolutions

def main : IO Unit := do
  IO.println (P58.solve 1 10 ⟨13120, P58Numerical.first_crossing.positive,
    P58Numerical.first_crossing.crossing⟩)
