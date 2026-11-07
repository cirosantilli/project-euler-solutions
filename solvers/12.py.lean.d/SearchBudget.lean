import solvers.«12.py.lean.d».SourceRuntime

namespace solvers.«12».runtimeCost
open ProjectEulerStatements.P12

/-- A prime within twice the square-root target keeps the complete source cost
quadratic in the first qualifying triangular index. -/
theorem search_budget_le (k bound : Nat)
    (upper : bound ≤ 2*(Nat.sqrt (firstIndex k+1)+1)) :
    searchWork k bound*firstIndex k+31 ≤ 55000*(firstIndex k+1)^2 := by
  have square := Nat.pow_le_pow_left (Nat.add_le_add_right upper 1) 2
  have root := ProjectEulerStatements.PrimeCacheBounds.sqrt_bound_square_le (firstIndex k+1)
  have small : (bound+1)^2 ≤ 32*(firstIndex k+1) := by omega
  have width : searchWork k bound ≤ 52000*(firstIndex k+1) := by
    dsimp only [searchWork, boundedFactorBudget, ensureBudget]
    nlinarith
  have scaled := Nat.mul_le_mul_right (firstIndex k) width
  calc
    searchWork k bound*firstIndex k+31 ≤ 52000*(firstIndex k+1)*firstIndex k+31 :=
      Nat.add_le_add_right scaled 31
    _ ≤ 55000*(firstIndex k+1)^2 := by nlinarith

end solvers.«12».runtimeCost
