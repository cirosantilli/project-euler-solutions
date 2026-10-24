import «.formalsw».solvers.«15.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p15.py»
import FormalSW
import ProjectEulerSolutions.Equiv.P15

namespace solvers.«15»
open FormalSW.PyAst FormalSW.PyAst.Calls
private abbrev program := Solvers.P15Ast.module

private def body : List Stmt :=
  match executableBody ((findFunctionIn program.body "binom").getD ([], [])).2 with
  | [_, _, _, .forIn _ _ body _, _] => body
  | _ => []

private theorem update_evaluates (n count i res : Nat) (bound : count ≤ n) (positive : 0 < i)
    (env : ScalarEnv) (getN : env "n" = some (.int n)) (getK : env "k" = some (.int count))
    (getI : env "i" = some (.int i)) (getRes : env "res" = some (.int res)) :
    Evaluates program env
      (.binOp (.binOp (.name "res") .mult
        (.binOp (.binOp (.name "n") .sub (.name "k")) .add (.name "i"))) .floorDiv (.name "i"))
      (.int ((res * (n - count + i) / i : Nat) : Int)) := by
  have sub : Evaluates program env (.binOp (.name "n") .sub (.name "k"))
      (.int ((n - count : Nat) : Int)) :=
    Evaluates.binary (Evaluates.lookup getN) (Evaluates.lookup getK)
      (by simp [scalarBinary, Int.natCast_sub bound])
  have add : Evaluates program env
      (.binOp (.binOp (.name "n") .sub (.name "k")) .add (.name "i"))
      (.int ((n - count + i : Nat) : Int)) :=
    Evaluates.binary sub (Evaluates.lookup getI) (by simp [scalarBinary])
  have product := Evaluates.binary (op := .mult) (value := .int ((res * (n - count + i) : Nat) : Int))
    (Evaluates.lookup getRes) add (by simp [scalarBinary])
  exact Evaluates.binary product (Evaluates.lookup getI)
    (by simp [scalarBinary, positive, ne_of_gt positive])

private theorem loop_result (n count i res : Nat) (bound : count ≤ n) (positive : 0 < i)
    (env : ScalarEnv) (getN : env "n" = some (.int n)) (getK : env "k" = some (.int count))
    (getRes : env "res" = some (.int res)) :
    BlockReturns program env [.rangeStepLoop "i" i (count + 1) 1 body [], .return (.name "res")]
      (.int (ProjectEulerSolutions.P15.binom.loop n count i res)) := by
  induction measure : count + 1 - i using Nat.strong_induction_on generalizing i res env with
  | h size ih =>
      rw [ProjectEulerSolutions.P15.binom.loop.eq_1]
      by_cases finished : i > count
      · rw [if_pos finished]
        apply BlockReturns.range_done (by simp [rangeContinues]; omega)
        exact BlockReturns.return_value (Evaluates.lookup getRes)
      · rw [if_neg finished]
        apply BlockReturns.range_step (by simp [rangeContinues]; omega)
        apply BlockReturns.assign (update_evaluates n count i res bound positive _
          (by simpa [ScalarEnv.set] using getN) (by simpa [ScalarEnv.set] using getK)
          (by simp [ScalarEnv.set]) (by simpa [ScalarEnv.set] using getRes))
        have next := ih (count + 1 - (i + 1)) (by omega) (i + 1) (res * (n - count + i) / i)
          (by omega) ((env.set "i" (.int i)).set "res" (.int ((res * (n - count + i) / i : Nat) : Int)))
          (by simpa [ScalarEnv.set] using getN) (by simpa [ScalarEnv.set] using getK)
          (by simp [ScalarEnv.set]) rfl
        simpa only [loopContinuation, show loopControl body = false from rfl, Bool.false_eq_true,
          if_false, Int.natCast_add, Int.natCast_one] using next

theorem binom.result (n k : Nat) :
    ReturnsValue program "binom" [.int n, .int k] (.int (ProjectEulerSolutions.P15.binom n k)) := by
  apply BlockReturns.function rfl rfl
  have condition : Evaluates program (argumentEnv
      ((findFunctionIn program.body "binom").getD ([], [])).1 [.int n, .int k])
      (.boolOp .or [.compare (.name "k") [(.lt, .constant (.int 0))],
        .compare (.name "k") [(.gt, .name "n")]]) (.bool (decide (k > n))) := by
    apply Evaluates.boolean_continue (first := .bool false)
      (Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant (by simp [scalarCompare, evalCmpOp])) rfl
    apply Evaluates.boolean_last
    exact Evaluates.compare (Evaluates.lookup rfl) (Evaluates.lookup rfl) (by simp [scalarCompare, evalCmpOp])
  by_cases large : k > n
  · apply BlockReturns.branch (value := .bool true) (by simpa [large] using condition)
    simp only [ProjectEulerSolutions.P15.binom, if_pos large]
    exact BlockReturns.return_value Evaluates.constant
  · apply BlockReturns.branch (value := .bool false) (by simpa [large] using condition)
    apply BlockReturns.assign (value := .int ((min k (n - k) : Nat) : Int))
    · apply Evaluates.builtin (.cons (Evaluates.lookup rfl) (.cons
        (Evaluates.binary (value := .int (n - k : Nat)) (Evaluates.lookup rfl)
          (Evaluates.lookup rfl) (by simp [scalarBinary, Int.natCast_sub (by omega : k ≤ n)])) .nil))
      simp [Primitive.apply, positionalMinimum, minimumFrom, scalarCompare, evalCmpOp]
      split <;> congr 3 <;> omega
    · apply BlockReturns.assign Evaluates.constant
      apply BlockReturns.for_range (values := [.int 1, .int ((min k (n - k) + 1 : Nat) : Int)])
      · exact .cons Evaluates.constant (.cons (Evaluates.binary
          (Evaluates.lookup (value := .int ((min k (n - k) : Nat) : Int)) (by simp [ScalarEnv.set]))
          Evaluates.constant (by simp [scalarBinary])) .nil)
      · rfl
      · have count : (if k < n - k then k else n - k) = min k (n - k) := by split_ifs <;> omega
        simpa only [ProjectEulerSolutions.P15.binom, if_neg large, count] using
          loop_result n (min k (n - k)) 1 1 (by omega) (by decide) _
            (by simp [ScalarEnv.set, argumentEnv]) (by simp [ScalarEnv.set]) (by simp [ScalarEnv.set])

theorem lattice_paths.result (n m : Nat) :
    ReturnsValue program "lattice_paths" [.int n, .int m]
      (.int (ProjectEulerSolutions.P15.latticePaths n m)) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  apply Evaluates.call (.cons (Evaluates.binary (value := .int (n + m : Nat))
    (Evaluates.lookup rfl) (Evaluates.lookup rfl) (by simp [scalarBinary]))
    (.cons (Evaluates.lookup rfl) .nil))
  exact binom.result (n + m) n

theorem solve.result (n : Nat) :
    ReturnsValue program "solve" [.int n] (.int (ProjectEulerSolutions.P15.solve n)) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  exact Evaluates.call (.cons (Evaluates.lookup rfl) (.cons (Evaluates.lookup rfl) .nil))
    (lattice_paths.result n n)

theorem solve.«Equiv(naive)» (n : Int) (condition : Condition program
    (fun name => if name = "n" then some (.int n) else none) Solvers.P15Ast.requires_solve_0) :
    (function program "solve" [.int n]).Equiv (function P15Ast.module "naive" [.int n]) := by
  have nonnegative : 0 ≤ n := by
    simpa [evalCmpOp] using Evaluates.condition_int_compare condition rfl
  lift n to Nat using nonnegative
  have reference := data.«project-euler-statements».data.py.project_euler_statements.p15.naive.result n
  rw [ProjectEulerSolutions.P15.equiv] at reference
  exact (solve.result n).equiv reference

end solvers.«15»
