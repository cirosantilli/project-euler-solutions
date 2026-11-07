import «.formalsw».solvers.«6.py.ast»
import data.«project-euler-statements».«.formalsw».data.py.project_euler_statements.«p6.py.ast»
import lang.py.FormalSW.CallsCost

/-! Execution-transition complexity for the actual Euler 6 Python sources.
Integer arithmetic is unit cost here; this is not a bit-complexity claim. -/
namespace solvers.«6».complexity
open FormalSW.PyAst
open FormalSW.PyAst.Calls
set_option maxRecDepth 4096
set_option maxHeartbeats 800000

/-- All inputs take exactly 51 interpreter transitions. -/
theorem optimized_exact (n : Nat) :
    ∃ value, FinishesIn (step Solvers.P6Ast.module)
      ⟨.invoke "solve" [.int n], []⟩ 51 (.ok value) := by
  refine ⟨.int (((n : Int) * (n + 1) / 2) * (n * (n + 1) / 2) -
    n * (n + 1) * (2 * n + 1) / 6), ?_⟩
  repeat first | refine FinishesIn.next rfl ?_ | exact FinishesIn.done rfl

private def env (n : Nat) : ScalarEnv := argumentEnv [⟨"n", some (.name "int"), none⟩] [.int n]
private def squares (n : Nat) : RangeSum :=
  ⟨"i", .binOp (.name "i") .mult (.name "i"), [], n + 1, 1, env n⟩
private def integers (n : Nat) : RangeSum :=
  ⟨"", .name "", [], n + 1, 1, env n⟩

private theorem squares_cost (n : Nat) (stack : List Frame) :
    ∃ answer, ReachesIn (step P6Ast.module) ⟨.invoke "sum_squares" [.int n], stack⟩
      (9 * n + 13) ⟨.value (.int answer), stack⟩ := by
  have item : ∀ i, Cost.EvaluatesIn P6Ast.module ((squares n).env.set "i" (.int i))
      (squares n).element 5 (.int (i * i)) := by
    intro i
    exact Cost.EvaluatesIn.binary (Cost.EvaluatesIn.lookup rfl) (Cost.EvaluatesIn.lookup rfl) rfl
  obtain ⟨answer, run⟩ := Cost.rangeSum P6Ast.module (squares n) (fun i => i * i) 5 rfl rfl
    item n 1 0 (by simp [squares, Int.add_comm]) stack
  have lead : ReachesIn (step P6Ast.module) ⟨.invoke "sum_squares" [.int n], stack⟩ 12
      ⟨.sumLoop (squares n) 1 0, stack⟩ := by
    repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  refine ⟨answer, ?_⟩
  have eq : 9 * n + 13 = 12 + ((5 + 4) * n + 1) := by omega
  rw [eq]
  exact lead.trans run

private theorem integers_cost (n : Nat) (stack : List Frame) :
    ∃ answer, ReachesIn (step P6Ast.module) ⟨.invoke "square_sum" [.int n], stack⟩
      (5 * n + 17) ⟨.value (.int answer), stack⟩ := by
  have item : ∀ i, Cost.EvaluatesIn P6Ast.module ((integers n).env.set "" (.int i))
      (integers n).element 1 (.int i) := fun _ => Cost.EvaluatesIn.lookup rfl
  let suffix : Frame := .binaryLeft .pow (.constant (.int 2)) (env n)
  obtain ⟨answer, run⟩ := Cost.rangeSum P6Ast.module (integers n) (fun i => i) 1 rfl rfl
    item n 1 0 (by simp [integers, Int.add_comm]) (suffix :: stack)
  have lead : ReachesIn (step P6Ast.module) ⟨.invoke "square_sum" [.int n], stack⟩ 13
      ⟨.sumLoop (integers n) 1 0, suffix :: stack⟩ := by
    repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  have tail : ReachesIn (step P6Ast.module) ⟨.value (.int answer), suffix :: stack⟩ 3
      ⟨.value (.int (answer ^ 2)), stack⟩ := by
    repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  refine ⟨answer ^ 2, ?_⟩
  have eq : 5 * n + 17 = (13 + ((1 + 4) * n + 1)) + 3 := by omega
  rw [eq]
  exact (lead.trans run).trans tail

/-- The reference includes both helper calls and both generator traversals. -/
theorem reference_exact (n : Nat) :
    ∃ value, FinishesIn (step P6Ast.module) ⟨.invoke "naive" [.int n], []⟩
      (14 * n + 42) (.ok value) := by
  let suffix : Frame := .binaryLeft .sub (.call (.name "sum_squares") [.name "n"]) (env n)
  obtain ⟨left, leftRun⟩ := integers_cost n [suffix]
  obtain ⟨right, rightRun⟩ := squares_cost n [.binaryRight .sub (.int left)]
  have lead : ReachesIn (step P6Ast.module) ⟨.invoke "naive" [.int n], []⟩ 6
      ⟨.invoke "square_sum" [.int n], [suffix]⟩ := by
    repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  have middle : ReachesIn (step P6Ast.module) ⟨.value (.int left), [suffix]⟩ 4
      ⟨.invoke "sum_squares" [.int n], [.binaryRight .sub (.int left)]⟩ := by
    repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  have tail : FinishesIn (step P6Ast.module)
      ⟨.value (.int right), [.binaryRight .sub (.int left)]⟩ 2 (.ok (.int (left - right))) :=
    .next rfl (.done rfl)
  refine ⟨.int (left - right), ?_⟩
  have eq : 14 * n + 42 = ((6 + (5 * n + 17)) + 4) + (9 * n + 13) + 2 := by omega
  rw [eq]
  exact (((lead.trans leftRun).trans middle).trans rightRun).finishes tail

def optimized (n : Nat) : Execution Constant := function Solvers.P6Ast.module "solve" [.int n]
def reference (n : Nat) : Execution Constant := function P6Ast.module "naive" [.int n]

/-- Constant interpreter cost for the optimized source, on every natural input. -/
theorem optimized_bigO : Execution.IsBigO optimized (fun _ => True) id (fun _ => 1) := by
  refine ⟨51, 0, fun n _ _ => ?_⟩
  obtain ⟨value, run⟩ := optimized_exact n
  exact ⟨51, .ok value, run, by simp⟩

/-- Linear interpreter cost for the reference source. -/
theorem reference_bigO : Execution.IsBigO reference (fun _ => True) id (fun n => n) := by
  refine ⟨56, 1, fun n _ large => ?_⟩
  obtain ⟨value, run⟩ := reference_exact n
  exact ⟨14 * n + 42, .ok value, run, by simp only [id_eq] at *; omega⟩

/-- The separation is strict: the reference cannot have a constant bound. -/
theorem reference_not_constant :
    ¬ Execution.IsBigO reference (fun _ => True) id (fun _ => 1) := by
  rintro ⟨constant, threshold, bounded⟩
  let n := constant + threshold + 1
  obtain ⟨cost, outcome, finished, small⟩ := bounded n trivial (by dsimp [n]; omega)
  obtain ⟨value, run⟩ := reference_exact n
  have same := finished.unique run
  simp only [Nat.mul_one] at small
  dsimp [n] at same
  omega

end solvers.«6».complexity

