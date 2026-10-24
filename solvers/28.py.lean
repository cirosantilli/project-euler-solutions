import «.formalsw».solvers.«28.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p28.py»
import ProjectEulerSolutions.Equiv.P28
import lang.py.FormalSW.Calls

namespace solvers.«28»
open FormalSW.PyAst FormalSW.PyAst.Calls
private abbrev program := Solvers.P28Ast.module
private abbrev reference := data.«project-euler-statements».data.py.project_euler_statements.p28.naive.result

private theorem condition_result (n : Int) (env : ScalarEnv)
    (lookup : env "n" = some (.int n)) :
    Evaluates program env Solvers.P28Ast.requires_solve_0
      (.bool (decide (1 ≤ n ∧ n % 2 = 1))) := by
  by_cases positive : 1 ≤ n
  · apply Evaluates.boolean_continue (first := .bool true)
    · exact Evaluates.compare (Evaluates.lookup lookup) Evaluates.constant
        (by simp [scalarCompare, evalCmpOp, positive])
    · rfl
    · apply Evaluates.boolean_last
      exact Evaluates.compare
        (Evaluates.binary (Evaluates.lookup lookup) Evaluates.constant rfl) Evaluates.constant
        (by simp [scalarCompare, evalCmpOp, positive])
  · simp only [positive, false_and, decide_false]
    apply Evaluates.boolean_stop (value := .bool false)
    · exact Evaluates.compare (Evaluates.lookup lookup) Evaluates.constant
        (by simp [scalarCompare, evalCmpOp, positive])
    · rfl

private def formula (m : Int) : Int :=
  1 + 16 * m * (m + 1) * (2 * m + 1) / 6 + 4 * m * (m + 1) / 2 + 4 * m

private theorem result_int (n : Int) (positive : 1 ≤ n) (odd : n % 2 = 1) :
    ReturnsValue program "solve" [.int n] (.int (formula ((n - 1) / 2))) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assertion (value := .bool true)
  · simpa [positive, odd] using condition_result n _ rfl
  · rfl
  · apply BlockReturns.assign (value := .int ((n - 1) / 2))
    · exact Evaluates.binary
        (Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant rfl) Evaluates.constant rfl
    · apply BlockReturns.assign Evaluates.constant
      apply BlockReturns.augAssign
        (value := .int (1 + 16 * ((n - 1) / 2) * ((n - 1) / 2 + 1) * (2 * ((n - 1) / 2) + 1) / 6))
      · repeat' first | apply Evaluates.binary | exact Evaluates.lookup rfl | exact Evaluates.constant | rfl
      · apply BlockReturns.augAssign
          (value := .int (1 + 16 * ((n - 1) / 2) * ((n - 1) / 2 + 1) * (2 * ((n - 1) / 2) + 1) / 6 +
            4 * ((n - 1) / 2) * ((n - 1) / 2 + 1) / 2))
        · repeat' first | apply Evaluates.binary | exact Evaluates.lookup rfl | exact Evaluates.constant | rfl
        · apply BlockReturns.augAssign (value := .int (formula ((n - 1) / 2)))
          · repeat' first | apply Evaluates.binary | exact Evaluates.lookup rfl | exact Evaluates.constant | rfl
          · exact BlockReturns.return_value (Evaluates.lookup rfl)

theorem solve.result (n : Nat) (positive : 1 ≤ n) (odd : n % 2 = 1) :
    ReturnsValue program "solve" [.int n] (.int (ProjectEulerSolutions.P28.solve n)) := by
  have difference : (n : Int) - 1 = ((n - 1 : Nat) : Int) := by omega
  have oddInt : (n : Int) % 2 = 1 := by exact_mod_cast odd
  have result := result_int n (by exact_mod_cast positive) oddInt
  rw [difference] at result
  simpa [formula, ProjectEulerSolutions.P28.solve] using result

private theorem domain (n : Int) (requires : Condition program
    (fun key => if key = "n" then some (.int n) else none)
    Solvers.P28Ast.requires_solve_0) : 1 ≤ n ∧ n % 2 = 1 := by
  have evaluated := (condition_result n _ rfl).condition_iff.mp requires
  simpa using evaluated

theorem solve.«Equiv(naive)» (n : Int) (requires : Condition program
    (fun key => if key = "n" then some (.int n) else none)
    Solvers.P28Ast.requires_solve_0) :
    (Calls.function program "solve" [.int n]).Equiv
      (Calls.function P28Ast.module "naive" [.int n]) := by
  obtain ⟨positive, odd⟩ := domain n requires
  lift n to Nat using (by omega : 0 ≤ n)
  have positiveNat : 1 ≤ n := by exact_mod_cast positive
  have optimized := solve.result n positiveNat (by exact_mod_cast odd)
  have naive := reference n positiveNat
  rw [ProjectEulerSolutions.P28.equiv] at naive
  exact optimized.equiv naive

theorem solve.«Supported» :
    (∀ n : Int, Condition program (fun key => if key = "n" then some (.int n) else none)
      Solvers.P28Ast.requires_solve_0 →
      (Calls.function program "solve" [.int n]).Supported) ∧
    (∀ n : Int, Condition program (fun key => if key = "n" then some (.int n) else none)
      Solvers.P28Ast.requires_solve_0 →
      (Calls.function P28Ast.module "naive" [.int n]).Supported) := by
  constructor
  · intro n requires
    obtain ⟨positive, odd⟩ := domain n requires
    exact (result_int n positive odd).returns.supported
  · intro n _
    exact (data.«project-euler-statements».data.py.project_euler_statements.p28.naive.«Returns» n).supported

end solvers.«28»
