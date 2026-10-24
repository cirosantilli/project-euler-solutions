import ProjectEulerStatements.P69
import ProjectEulerStatements.Lib.TotientPrimorialScan

set_option linter.style.header false

/-! The consecutive-prime product scan stops at a bound derived from its input. -/
namespace ProjectEulerSolutions.P69

def solve (limit : Int) : Nat := ProjectEulerStatements.TotientMaximum.optimizedResult limit

end ProjectEulerSolutions.P69
