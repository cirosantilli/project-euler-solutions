import ProjectEulerSolutions.P73

set_option linter.style.header false

/-! Range pruning preserves all fractions in every interval with positive endpoint denominators. -/
namespace ProjectEulerSolutions.P73

theorem equiv (limit lowerNum lowerDen upperNum upperDen : Int)
    (lowerPositive : 0 < lowerDen) (upperPositive : 0 < upperDen) :
    ProjectEulerStatements.P73.naive limit lowerNum lowerDen upperNum upperDen =
      solve limit lowerNum lowerDen upperNum upperDen :=
  (solve_eq_result limit ⟨lowerNum, lowerDen, upperNum, upperDen⟩
    ⟨lowerPositive, upperPositive⟩).symm

end ProjectEulerSolutions.P73
