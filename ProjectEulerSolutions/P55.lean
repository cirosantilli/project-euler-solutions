import ProjectEulerStatements.P55
import ProjectEulerSolutions.Lib.Lychrel

namespace ProjectEulerSolutions.P55

/-- Reverse-and-add with a structurally decreasing iteration budget. -/
def isLychrelCandidate (n maxIters : Nat) : Bool := Lychrel.candidate n maxIters

def solve (limit maxIters : Nat) : Nat := Lychrel.count limit maxIters

end ProjectEulerSolutions.P55
