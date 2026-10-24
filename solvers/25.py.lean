import «.formalsw».solvers.«25.py.ast»
import ProjectEulerStatements.Lib.FibonacciDigits
import lang.py.FormalSW.Calls
import data.«project-euler-statements».data.py.project_euler_statements.«p25.py»

namespace solvers.«25»
open FormalSW.PyAst FormalSW.PyAst.Calls
open ProjectEulerStatements.FibonacciDigits

private abbrev program := Solvers.P25Ast.module
private def loop : List Stmt :=
  (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 4

private theorem decimal_length (index : Nat) (positive : 0 < index) (env : ScalarEnv)
    (lookup : env "b" = some (.int (Nat.fib index))) :
    Evaluates program env (.builtinCall .len [.builtinCall .str [.name "b"]])
      (.int (ProjectEulerStatements.P25.numDigits (Nat.fib index))) := by
  apply Evaluates.builtin
    (.cons (Evaluates.builtin (.cons (Evaluates.lookup lookup) .nil) rfl) .nil)
  have length := fib_string_length index positive
  simpa [Primitive.apply, Calls.length, Strings.convert] using length

/-- The iterative source preserves consecutive Fibonacci numbers and advances toward the least index. -/
private theorem loop_result (digits index : Nat) (positive : 0 < index)
    (bound : index + 1 ≤ first digits) (env : ScalarEnv)
    (getDigits : env "digits" = some (.int digits))
    (getA : env "a" = some (.int (Nat.fib index)))
    (getB : env "b" = some (.int (Nat.fib (index + 1))))
    (getN : env "n" = some (.int (index + 1))) :
    BlockReturns program env loop (.int (first digits)) := by
  induction measure : first digits - (index + 1) using Nat.strong_induction_on generalizing index env with
  | h measure ih =>
      by_cases more : index + 1 < first digits
      · have length := before_first digits (index + 1) more
        apply BlockReturns.while_loop (value := .bool true)
        · exact Evaluates.compare (decimal_length (index + 1) (by omega) env getB)
            (Evaluates.lookup getDigits) (by simp [scalarCompare, evalCmpOp, length])
        · apply BlockReturns.unpack (names := ["a", "b"])
            (results := [.int (Nat.fib (index + 1)), .int (Nat.fib (index + 2))]) rfl rfl
          · exact .cons (Evaluates.lookup getB) (.cons
              (Evaluates.binary (Evaluates.lookup getA) (Evaluates.lookup getB)
                (by simp [scalarBinary, Nat.fib_add_two])) .nil)
          · apply BlockReturns.augAssign (value := .int (index + 2))
            · exact Evaluates.binary (Evaluates.lookup (by simpa [assignNames, ScalarEnv.set] using getN))
                Evaluates.constant (by simp [scalarBinary]; omega)
            · apply ih (first digits - ((index + 1) + 1)) (by omega) (index + 1) (by omega) (by omega)
              all_goals simp [assignNames, ScalarEnv.set, getDigits] <;> omega
      · have same : index + 1 = first digits := by omega
        have reached : digits ≤ ProjectEulerStatements.P25.numDigits (Nat.fib (index + 1)) := by
          rw [same]; exact first_spec digits
        apply BlockReturns.while_loop (value := .bool false)
        · exact Evaluates.compare (decimal_length (index + 1) (by omega) env getB)
            (Evaluates.lookup getDigits) (by simp [scalarCompare, evalCmpOp]; omega)
        · have castSame : (index : Int) + 1 = (first digits : Int) := by exact_mod_cast same
          exact BlockReturns.return_value (Evaluates.lookup (by simpa only [castSame] using getN))

/-- Exact result of the optimized Python function, including its one-digit special case. -/
theorem solve.result (digits : Nat) (positive : 0 < digits) :
    ReturnsValue program "solve" [.int digits] (.int (first digits)) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assertion (value := .bool true)
  · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
      (by simp [scalarCompare, evalCmpOp]; omega)
  · rfl
  · apply BlockReturns.unpack (names := ["a", "b"]) (results := [.int 1, .int 1]) rfl rfl
    · exact .cons Evaluates.constant (.cons Evaluates.constant .nil)
    · by_cases one : digits = 1
      · subst digits
        apply BlockReturns.branch (value := .bool true)
        · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant rfl
        · rw [first_one]
          exact BlockReturns.return_value Evaluates.constant
      · apply BlockReturns.branch (value := .bool false)
        · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
            (by simp [scalarCompare, evalCmpOp, one])
        · apply BlockReturns.assign Evaluates.constant
          have initial : 2 ≤ first digits := by
            have pos := first_positive digits positive
            have spec := first_spec digits
            have small : ProjectEulerStatements.P25.numDigits (Nat.fib 1) = 1 := by decide
            by_contra absent
            have equal : first digits = 1 := by omega
            rw [equal, small] at spec
            omega
          exact loop_result digits 1 (by decide) initial _ rfl rfl rfl rfl

theorem solve.«Returns» (digits : Int)
    (requires : Calls.Condition program
      (fun key => if key = "digits" then some (.int digits) else none)
      Solvers.P25Ast.requires_solve_0) :
    (Calls.function program "solve" [.int digits]).Returns := by
  have positive : 1 ≤ digits := by
    simpa [evalCmpOp] using Evaluates.condition_int_compare requires rfl
  lift digits to Nat using (by omega : 0 ≤ digits)
  exact (solve.result digits (by omega)).returns


theorem solve.«Equiv(naive)» (digits : Int)
    (requires : Calls.Condition program
      (fun key => if key = "digits" then some (.int digits) else none)
      Solvers.P25Ast.requires_solve_0) :
    (Calls.function program "solve" [.int digits]).Equiv
      (MemoHeapCalls.scalarFunction P25Ast.module "naive" [.int digits]) := by
  have positive : 1 ≤ digits := by
    simpa [evalCmpOp] using Evaluates.condition_int_compare requires rfl
  lift digits to Nat using (by omega : 0 ≤ digits)
  have optimized := (solve.result digits (by omega)).finishes
  obtain ⟨heap, result⟩ := data.«project-euler-statements».data.py.project_euler_statements.p25.naive.result digits []
  have reference := MemoHeapCalls.ReturnsValue.scalar_finishes (values := [.int digits]) result
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem solve.«Supported» :
    (∀ digits : Int, Calls.Condition program
      (fun key => if key = "digits" then some (.int digits) else none)
      Solvers.P25Ast.requires_solve_0 →
      (Calls.function program "solve" [.int digits]).Supported) ∧
    (∀ digits : Int, Calls.Condition program
      (fun key => if key = "digits" then some (.int digits) else none)
      Solvers.P25Ast.requires_solve_0 →
      (MemoHeapCalls.scalarFunction P25Ast.module "naive" [.int digits]).Supported) := by
  constructor
  · intro digits requires
    exact (solve.«Returns» digits requires).supported
  · intro digits requires
    have positive : 1 ≤ digits := by
      simpa [evalCmpOp] using Evaluates.condition_int_compare requires rfl
    lift digits to Nat using (by omega : 0 ≤ digits)
    obtain ⟨heap, result⟩ := data.«project-euler-statements».data.py.project_euler_statements.p25.naive.result digits []
    exact (MemoHeapCalls.ReturnsValue.scalar_returns result).supported

end solvers.«25»
