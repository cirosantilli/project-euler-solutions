import «.formalsw».solvers.«28.py.ast»
import lang.py.FormalSW.Calls

namespace solvers.«28»
open FormalSW.PyAst FormalSW.PyAst.Calls
abbrev program := Solvers.P28Ast.module

theorem condition_result (n : Int) (env : ScalarEnv)
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

def formula (m : Int) : Int :=
  1 + 16 * m * (m + 1) * (2 * m + 1) / 6 + 4 * m * (m + 1) / 2 + 4 * m

theorem domain (n : Int) (requires : Condition program
    (fun key => if key = "n" then some (.int n) else none)
    Solvers.P28Ast.requires_solve_0) : 1 ≤ n ∧ n % 2 = 1 := by
  have evaluated := (condition_result n _ rfl).condition_iff.mp requires
  simpa using evaluated

end solvers.«28»
