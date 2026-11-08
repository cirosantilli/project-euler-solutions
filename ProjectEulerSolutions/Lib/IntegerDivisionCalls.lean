import ProjectEulerSolutions.Lib.MutableIntegerMatrixAccess

/-! Python integer floor division for either sign of a nonzero divisor. -/
namespace ProjectEulerSolutions.IntegerDivisionCalls
open FormalSW.PyAst
set_option Elab.async false

theorem floor_divide (numerator divisor : Int) (heap : Mutable.Heap) (nonzero : divisor ≠ 0) :
    Mutable.binary .floorDiv (.scalar (.int numerator)) (.scalar (.int divisor)) heap =
      some (.ok (.scalar (.int (numerator.fdiv divisor))), heap) := by
  by_cases positive : 0 < divisor
  · simp [Mutable.binary, scalarBinary, nonzero, positive,
      Int.fdiv_eq_ediv_of_nonneg numerator (by omega : 0 ≤ divisor)]
  · have quotient : numerator.fdiv divisor = (-numerator) / (-divisor) := by
      rw [← Int.neg_fdiv_neg, Int.fdiv_eq_ediv_of_nonneg _ (by omega : 0 ≤ -divisor)]
    simp [Mutable.binary, scalarBinary, nonzero, positive, quotient]

end ProjectEulerSolutions.IntegerDivisionCalls
