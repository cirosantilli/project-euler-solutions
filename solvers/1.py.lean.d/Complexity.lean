import «.formalsw».solvers.«1.py.ast»
import data.«project-euler-statements».«.formalsw».data.py.project_euler_statements.«p1.py.ast»
import lang.py.FormalSW.CallsCost

/-! Source-linked transition costs for Euler 1 after the frontend's explicit
`canonical-leading-contracts-v1` Nagini erasure. Contract arguments are ghost
syntax and are not evaluated; this is not CPython executing the contract library.
The fresh-source frontend checks canonical imports and rejects annotation-name
rebinding or runtime use. Arithmetic uses unit-cost unbounded integers. -/
namespace solvers.«1».complexity
open FormalSW.PyAst
open FormalSW.PyAst.Calls
set_option maxRecDepth 4096
set_option maxHeartbeats 800000

/-- Includes all three calls to `sum_of_multiples`. -/
theorem optimized_exact (n : Nat) :
    ∃ value, FinishesIn (step Solvers.P1Ast.module)
      ⟨.invoke "solve" [.int n], []⟩ 109 (.ok value) := by
  refine ⟨.int (3 * (((n : Int) - 1) / 3) * (((n - 1) / 3) + 1) / 2 +
    5 * ((n - 1) / 5) * (((n - 1) / 5) + 1) / 2 -
    15 * ((n - 1) / 15) * (((n - 1) / 15) + 1) / 2), ?_⟩
  repeat first | refine FinishesIn.next rfl ?_ | exact FinishesIn.done rfl

private def env (n : Nat) : ScalarEnv := argumentEnv [⟨"n", some (.name "int"), none⟩] [.int n]

private def answerExpr : Expr :=
  .ifExp (.boolOp .or
    [.compare (.binOp (.name "m") .mod (.constant (.int 3))) [(.eq, .constant (.int 0))],
     .compare (.binOp (.name "m") .mod (.constant (.int 5))) [(.eq, .constant (.int 0))]])
    (.binOp (.name "s") .add (.name "m")) (.name "s")

private def continuation (n : Nat) : Frame :=
  .assign "s" [.return answerExpr] ((env (n + 1)).set "m" (.int n))

private theorem reference_prefix (n : Nat) (stack : List Frame) :
    ReachesIn (step P1Ast.module) ⟨.invoke "naive" [.int (n + 1 : Nat)], stack⟩ 19
      ⟨.invoke "naive" [.int n], continuation n :: stack⟩ := by
  have nonzero : (n : Int) + 1 ≠ 0 := by omega
  repeat first
    | exact ReachesIn.refl
    | refine ReachesIn.next rfl ?_
    | refine ReachesIn.next (by simp [step, scalarCompare, evalCmpOp, nonzero]) ?_
  simp only [Int.natCast_add, Int.natCast_one, Int.add_sub_cancel, List.nil_append]
  exact .refl

private theorem reference_suffix (n : Nat) (value : Int) (stack : List Frame) :
    ∃ cost answer, 20 ≤ cost ∧ cost ≤ 30 ∧
      ReachesIn (step P1Ast.module) ⟨.value (.int value), continuation n :: stack⟩ cost
        ⟨.value (.int answer), stack⟩ := by
  by_cases three : (n : Int) % 3 = 0
  · refine ⟨20, value + n, by omega, by omega, ?_⟩
    repeat first
      | exact ReachesIn.refl
      | refine ReachesIn.next rfl ?_
    simp only [evalCmpOp, three, decide_true]
    repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  · by_cases five : (n : Int) % 5 = 0
    · refine ⟨30, value + n, by omega, by omega, ?_⟩
      repeat first
        | exact ReachesIn.refl
        | refine ReachesIn.next rfl ?_
      simp only [evalCmpOp, three, decide_false]
      repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
      simp only [evalCmpOp, five, decide_true]
      repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
    · refine ⟨26, value, by omega, by omega, ?_⟩
      repeat first
        | exact ReachesIn.refl
        | refine ReachesIn.next rfl ?_
      simp only [evalCmpOp, three, decide_false]
      repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
      simp only [evalCmpOp, five, decide_false]
      repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_

private theorem reference_run (n : Nat) (stack : List Frame) :
    ∃ cost answer, 39 * n + 10 ≤ cost ∧ cost ≤ 49 * n + 10 ∧
      ReachesIn (step P1Ast.module) ⟨.invoke "naive" [.int n], stack⟩ cost
        ⟨.value (.int answer), stack⟩ := by
  induction n generalizing stack with
  | zero =>
      refine ⟨10, 0, by omega, by omega, ?_⟩
      repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  | succ n ih =>
      obtain ⟨cost, value, lower, upper, recursive⟩ := ih (continuation n :: stack)
      obtain ⟨tailCost, answer, tailLower, tailUpper, tail⟩ := reference_suffix n value stack
      exact ⟨19 + cost + tailCost, answer, by omega, by omega,
        ((reference_prefix n stack).trans recursive).trans tail⟩

/-- Short circuiting changes exact costs, but every recursive level costs 39–49. -/
theorem reference_bounds (n : Nat) :
    ∃ cost value, 39 * n + 11 ≤ cost ∧ cost ≤ 49 * n + 11 ∧
      FinishesIn (step P1Ast.module) ⟨.invoke "naive" [.int n], []⟩ cost (.ok value) := by
  obtain ⟨cost, value, lower, upper, run⟩ := reference_run n []
  exact ⟨cost + 1, .int value, by omega, by omega, run.finishes (.done rfl)⟩

def optimized (n : Nat) : Execution Constant := function Solvers.P1Ast.module "solve" [.int n]
def reference (n : Nat) : Execution Constant := function P1Ast.module "naive" [.int n]

theorem optimized_bigO : Execution.IsBigO optimized (fun _ => True) id (fun _ => 1) := by
  refine ⟨109, 0, fun n _ _ => ?_⟩
  obtain ⟨value, run⟩ := optimized_exact n
  exact ⟨109, .ok value, run, by simp⟩

theorem reference_bigO : Execution.IsBigO reference (fun _ => True) id (fun n => n) := by
  refine ⟨60, 1, fun n _ large => ?_⟩
  obtain ⟨cost, value, _, upper, run⟩ := reference_bounds n
  exact ⟨cost, .ok value, run, by simp only [id_eq] at *; omega⟩

theorem reference_not_constant :
    ¬ Execution.IsBigO reference (fun _ => True) id (fun _ => 1) := by
  rintro ⟨constant, threshold, bounded⟩
  let n := constant + threshold + 1
  obtain ⟨cost, outcome, finished, small⟩ := bounded n trivial (by dsimp [n]; omega)
  obtain ⟨actual, value, lower, _, run⟩ := reference_bounds n
  have same := finished.unique run
  simp only [Nat.mul_one] at small
  dsimp [n] at lower
  omega

end solvers.«1».complexity
