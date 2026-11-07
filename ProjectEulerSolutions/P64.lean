import ProjectEulerSolutions.Lib.SqrtPeriodCounts

namespace ProjectEulerSolutions.P64

/-- Count odd continued-fraction periods over the same finite range as Python. -/
def solve (limit : Nat) : Nat := SqrtPeriodCounts.answer limit

end ProjectEulerSolutions.P64
