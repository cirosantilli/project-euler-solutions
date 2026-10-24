import «.formalsw».solvers.«1.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p1.py»
import lang.py.FormalSW.Calls
import ProjectEulerSolutions.Equiv.P1

namespace solvers.«1»
open FormalSW.PyAst FormalSW.PyAst.Calls
open ProjectEulerSolutions.P1

private theorem sum_of_multiples_result (m n : Nat) (hm : 0 < m) :
    ReturnsValue Solvers.P1Ast.module "sum_of_multiples" [.int m, .int n]
      (.int (sumOfMultiples m n)) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := .int (n / m))
  · apply Evaluates.binary (Evaluates.lookup rfl) (Evaluates.lookup rfl)
    simp [scalarBinary, hm.ne', hm]
  · apply BlockReturns.return_value
    apply Evaluates.binary (av := .int (m * (n / m) * (n / m + 1))) (bv := .int 2)
    · apply Evaluates.binary (av := .int (m * (n / m))) (bv := .int (n / m + 1))
      · exact Evaluates.binary (Evaluates.lookup rfl) (Evaluates.lookup rfl)
          (by simp [scalarBinary])
      · exact Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant
          (by simp [scalarBinary])
      · simp [scalarBinary]
    · exact Evaluates.constant
    · simp [scalarBinary, sumOfMultiples]

private theorem solve_result (n : Nat) (hn : 1 < n) :
    ReturnsValue Solvers.P1Ast.module "solve" [.int n] (.int (solve n)) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.augAssign (value := .int ↑(n - 1))
  · apply Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant
    simp [scalarBinary, Int.natCast_sub (by omega : 1 ≤ n)]
  · apply BlockReturns.return_value
    apply Evaluates.binary
      (av := .int (sumOfMultiples 3 (n - 1) + sumOfMultiples 5 (n - 1)))
      (bv := .int (sumOfMultiples 15 (n - 1)))
    · apply Evaluates.binary
        (av := .int (sumOfMultiples 3 (n - 1))) (bv := .int (sumOfMultiples 5 (n - 1)))
      · exact Evaluates.call (.cons Evaluates.constant (.cons (Evaluates.lookup rfl) .nil))
          (sum_of_multiples_result 3 (n - 1) (by decide))
      · exact Evaluates.call (.cons Evaluates.constant (.cons (Evaluates.lookup rfl) .nil))
          (sum_of_multiples_result 5 (n - 1) (by decide))
      · simp [scalarBinary]
    · exact Evaluates.call (.cons Evaluates.constant (.cons (Evaluates.lookup rfl) .nil))
        (sum_of_multiples_result 15 (n - 1) (by decide))
    · have hsub : sumOfMultiples 15 (n - 1) ≤
          sumOfMultiples 3 (n - 1) + sumOfMultiples 5 (n - 1) :=
        le_trans (sumOfMultiples_le_of_dvd (by decide : 0 < 3) (by decide : 0 < 15)
          (by decide : 3 ∣ 15) (n - 1)) (Nat.le_add_right _ _)
      simp [scalarBinary, solve, Int.natCast_sub hsub]

theorem solve.«Equiv(naive)» (n : Int) (hp : Calls.Condition Solvers.P1Ast.module
      (fun key => if key = "n" then some (.int n) else none) Solvers.P1Ast.requires_solve_0) :
    (Calls.function Solvers.P1Ast.module "solve" [.int n]).Equiv
      (Calls.function P1Ast.module "naive" [.int n]) := by
  have hn : n > 1 := by
    simpa [evalCmpOp] using Evaluates.condition_int_compare hp rfl
  lift n to Nat using (by omega : 0 ≤ n)
  have hn : 1 < n := by omega
  have naive_result := data.«project-euler-statements».data.py.project_euler_statements.«p1».naive.result n
  rw [ProjectEulerSolutions.P1.equiv] at naive_result
  exact (solve_result n hn).equiv naive_result

end solvers.«1»
