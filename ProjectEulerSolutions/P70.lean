import ProjectEulerStatements.P70
import ProjectEulerSolutions.Lib.TotientArrayScanCorrect

set_option linter.style.header false

/-! The array-based totient-permutation search uses its signed inclusive input bound. -/
namespace ProjectEulerSolutions.P70

abbrev computePhiLinear := TotientArraySieve.computePhiLinear
abbrev digitSignature := TotientArrayScan.digitSignature
abbrev solveCore := TotientArrayScan.solveCore

def solve (limit : Int) : Nat := TotientArrayScan.solve limit

theorem solve_eq_reference (limit : Int) :
    solve limit = ProjectEulerStatements.TotientPermutation.result limit :=
  TotientArrayScan.solve_correct limit

end ProjectEulerSolutions.P70
