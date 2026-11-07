import ProjectEulerSolutions.P27
namespace ProjectEulerSolutions.P27

/-- Python's zero-argument API uses |a| < 1000 and |b| ≤ 1000. -/
theorem equiv : ProjectEulerStatements.P27.naive 999 1000 = solve :=
  P27Search.reference_answer.trans P27Search.optimized_answer.symm
end ProjectEulerSolutions.P27
