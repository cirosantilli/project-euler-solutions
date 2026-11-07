import ProjectEulerSolutions.Lib.SqrtConvergents

namespace ProjectEulerSolutions.P57

/-- The Python recurrence, with the finite range represented by a decreasing count. -/
def countSqrt2Convergents (limit : Nat) : Nat :=
  SqrtConvergents.loop limit 3 2 0

def solve (limit : Nat) : Nat := countSqrt2Convergents limit

end ProjectEulerSolutions.P57
