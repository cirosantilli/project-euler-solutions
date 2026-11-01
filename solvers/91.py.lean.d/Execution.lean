import solvers.«91.py.lean.d».Loops

namespace solvers.«91».solve
open FormalSW.PyAst Calls ProjectEulerSolutions.LatticeRightTriangles
set_option Elab.async false

theorem positive_result (n : Nat) (positive : 0 < n) :
    ReturnsValue program "solve" [number n] (number (optimizedCount n)) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (value := .bool false)
  · apply Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
    simp [scalarCompare, evalCmpOp, Nat.ne_of_gt positive]
  apply BlockReturns.assign (value := number (n * n))
  · exact Evaluates.binary (Evaluates.lookup rfl) (Evaluates.lookup rfl) (by simp [scalarBinary, number])
  apply BlockReturns.for_range (values := [number (n + 1)])
  · exact .cons (Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant (by simp [scalarBinary, number])) .nil
  · rfl
  apply rows_result n 0 (n * n) _ rfl rfl
  intro after _ getTotal
  apply BlockReturns.return_value
  apply Evaluates.lookup
  simpa only [rowsSum, rowSum, ← Nat.range_succ_eq_Icc_zero, optimizedCount] using getTotal

/-- Nonpositive bounds return before the loops; all positive bounds execute the complete grid traversal. -/
theorem result (n : Int) :
    ReturnsValue program "solve" [.int n] (number (optimizedCount n.toNat)) := by
  by_cases positive : 0 < n
  · have cast : (n.toNat : Int) = n := Int.toNat_of_nonneg (by omega)
    simpa only [number, cast] using positive_result n.toNat (by omega)
  · have zero : n.toNat = 0 := by omega
    have count : optimizedCount n.toNat = 0 := by simp [zero, optimizedCount, pointContribution]
    rw [count]
    apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (value := .bool true)
    · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
        (by simp [scalarCompare, evalCmpOp, show n ≤ 0 by omega])
    exact BlockReturns.return_value Evaluates.constant

end solvers.«91».solve
