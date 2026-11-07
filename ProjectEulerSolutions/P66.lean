import ProjectEulerStatements.P66
import ProjectEulerStatements.Lib.PellMinimalConvergent

namespace ProjectEulerSolutions.P66
open ProjectEulerStatements

/-- Search continued-fraction convergents for the first positive Pell solution. -/
def minimalPellX (d : Nat) : Nat :=
  if square : Nat.sqrt d * Nat.sqrt d = d then 0
  else (PellConvergents.convergents d (PellConvergents.firstHit d square)).currentP.toNat

def solve (limit : Nat) : Nat :=
  ((List.range' 2 (limit - 1)).foldl (fun (best : Nat × Nat) d =>
    let x := minimalPellX d
    if best.2 < x then (d, x) else best) (0, 0)).1

end ProjectEulerSolutions.P66
