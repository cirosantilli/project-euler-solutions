import «.formalsw».solvers.«5.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p5.py»
import FormalSW
import ProjectEulerSolutions.Equiv.P5

namespace solvers.«5»
open FormalSW.PyAst FormalSW.PyAst.Calls
private abbrev program := Solvers.P5Ast.module

private def body : List Stmt :=
  match executableBody ((findFunctionIn program.body "solve").getD ([], [])).2 with
  | [_, .forIn _ _ body _, _] => body
  | _ => []

/-- Division happens before multiplication in the optimized source. -/
private theorem update_evaluates (acc x : Nat) (positive : 0 < x) (env : ScalarEnv)
    (getAcc : env "ans" = some (.int acc)) (getX : env "x" = some (.int x)) :
    Evaluates program env
      (.binOp (.binOp (.name "ans") .floorDiv (.call (.name "gcd") [.name "ans", .name "x"]))
        .mult (.name "x")) (.int (ProjectEulerSolutions.P5.lcm acc x)) := by
  apply Evaluates.binary (av := .int ((acc / Nat.gcd acc x : Nat) : Int))
  · apply Evaluates.binary (Evaluates.lookup getAcc) (bv := .int (Nat.gcd acc x))
    · apply Evaluates.library_call (.cons (Evaluates.lookup getAcc) (.cons (Evaluates.lookup getX) .nil)) rfl
      change some (Math.gcdCall [.int acc, .int x]) = _
      simpa using congrArg some (Math.gcdCall_pair (acc : Int) (x : Int))
    · simp [scalarBinary, positive, ne_of_gt positive]
  · exact Evaluates.lookup getX
  · simp [scalarBinary, ProjectEulerSolutions.P5.lcm]

private theorem loop_result (n x acc : Nat) (positive : 0 < x) (env : ScalarEnv)
    (getAcc : env "ans" = some (.int acc)) :
    BlockReturns program env [.rangeStepLoop "x" x (n + 1) 1 body [], .return (.name "ans")]
      (.int (ProjectEulerSolutions.P5.go n x acc)) := by
  induction measure : n + 1 - x using Nat.strong_induction_on generalizing x acc env with
  | h size ih =>
      rw [ProjectEulerSolutions.P5.go.eq_1]
      by_cases finished : x > n
      · rw [if_pos finished]
        apply BlockReturns.range_done (by simp [rangeContinues]; omega)
        exact BlockReturns.return_value (Evaluates.lookup getAcc)
      · rw [if_neg finished]
        apply BlockReturns.range_step (by simp [rangeContinues]; omega)
        apply BlockReturns.assign (update_evaluates acc x positive _
          (by simpa [ScalarEnv.set] using getAcc) (by simp [ScalarEnv.set]))
        have next := ih (n + 1 - (x + 1)) (by omega) (x + 1) (ProjectEulerSolutions.P5.lcm acc x)
          (by omega) ((env.set "x" (.int x)).set "ans" (.int (ProjectEulerSolutions.P5.lcm acc x)))
          (by simp [ScalarEnv.set]) rfl
        simpa [loopContinuation, body, Int.natCast_add] using next

theorem solve.result (n : Nat) :
    ReturnsValue program "solve" [.int n] (.int (ProjectEulerSolutions.P5.solve n)) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_range (values := [.int 2, .int ((n + 1 : Nat) : Int)])
  · exact .cons Evaluates.constant (.cons
      (Evaluates.binary (Evaluates.lookup (value := .int n) (by simp [ScalarEnv.set, argumentEnv]))
        Evaluates.constant (by simp [scalarBinary])) .nil)
  · rfl
  · exact loop_result n 2 1 (by decide) _ rfl

theorem solve.«Equiv(naive)» (n : Int) (hp : Condition program
    (fun key => if key = "n" then some (.int n) else none) Solvers.P5Ast.requires_solve_0) :
    (function program "solve" [.int n]).Equiv (function P5Ast.module "naive" [.int n]) := by
  have nonnegative : 0 ≤ n := by
    simpa [evalCmpOp] using Evaluates.condition_int_compare hp rfl
  lift n to Nat using nonnegative
  have reference := data.«project-euler-statements».data.py.project_euler_statements.p5.naive.result n
  rw [ProjectEulerSolutions.P5.equiv] at reference
  exact (solve.result n).equiv reference

end solvers.«5»
