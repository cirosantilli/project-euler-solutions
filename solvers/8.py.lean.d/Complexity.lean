import data.«project-euler-statements».data.py.project_euler_statements.«p8.py»
import lang.py.FormalSW.AllocationCost
import solvers.«8.py.lean.d».ZeroComplexity
import Mathlib.Tactic.Linarith

/-! A source-linked quadratic allocation lower bound for Euler 8's reference,
even at fixed window width one. The cost model charges transitions and newly
allocated list slots; it does not pretend Python slices are constant cost. -/
namespace solvers.«8».complexity
open FormalSW.PyAst
open data.«project-euler-statements».data.py.project_euler_statements.p8
open HeapCalls.AllocationCost

private def ints (xs : List Int) : List Mutable.Value := xs.map fun x => .scalar (.int x)

theorem windows_one_length (xs : List Int) : (windowValues 1 xs).length = xs.length := by
  induction xs with
  | nil => rfl
  | cons head tail ih =>
      have enough : ¬ ((head :: tail).length : Int) < 1 := by simp
      rw [windowValues, if_neg enough]
      simp only [List.length_cons, ih]

/-- Prefix/suffix copies, singleton product lists and concatenations together. -/
theorem allocations_one (xs : List Int) : windowAllocations 1 xs = xs.length ^ 2 + 2 * xs.length := by
  induction xs with
  | nil => rfl
  | cons head tail ih =>
      have enough : ¬ ((head :: tail).length : Int) < 1 := by simp
      simp only [windowAllocations, if_neg enough, windows_one_length, ih]
      have sliceLength : FormalSW.Py.Builtin.Slice.prefixLength (head :: tail).length 1 = 1 := by
        change FormalSW.Py.Builtin.Slice.prefixLength _ (↑(1 : Nat)) = 1
        rw [FormalSW.Py.Builtin.Slice.prefixLength_nonnegative]
        simp
      rw [sliceLength]
      simp only [List.take_succ_cons, List.take_zero, List.length_cons, List.length_nil]
      nlinarith

/-- Fixed positive width keeps the lower bound genuinely about list length. -/
def reference (digits : List Int) : Execution Mutable.Completion :=
  (HeapCalls.call P8Ast.module "naive" [.ref 0, .scalar (.int 1)] [.list (ints digits)]).execution

def weights (digits : List Int) : (reference digits).State → Nat :=
  charge P8Ast.module 1

/-- Every terminating run of the actual reference pays at least quadratic work. -/
theorem reference_lower (digits : List Int)
    (run : FinishesWith (reference digits).step (weights digits)
      (reference digits).initial cost outcome) :
    digits.length ^ 2 + 2 * digits.length ≤ cost := by
  obtain ⟨last, _, allocated, executed⟩ := naive.result_allocations digits 1 [.list (ints digits)] 0 rfl
  have same := run.finishes.unique executed.finishes
  subst outcome
  have bound := allocated_le run
  change listSlots last ≤ listSlots [.list (ints digits)] + cost at bound
  rw [allocated, allocations_one] at bound
  omega

/-- The reference's list copying rules out every eventual linear bound. -/
theorem reference_not_linear :
    ¬ Execution.WeightedBigO reference weights (fun _ => True) List.length (fun n => n) := by
  rintro ⟨constant, threshold, bounded⟩
  let n := constant + threshold + 1
  let digits : List Int := List.replicate n 1
  have length : digits.length = n := by simp [digits]
  obtain ⟨cost, outcome, run, upper⟩ := bounded digits trivial (by rw [length]; dsimp [n]; omega)
  have lower := reference_lower digits run
  rw [length] at lower upper
  have large : constant < n := by dsimp [n]; omega
  have positive : 0 < n := by omega
  nlinarith

def referenceZeros (n : Nat) : Execution Mutable.Completion := reference (List.replicate n 0)
def referenceZeroWeights (n : Nat) : (referenceZeros n).State → Nat := weights (List.replicate n 0)

/-- The same zero-list family has a linear optimized trace, but no linear
reference trace, under exactly the same transition-plus-allocation charges. -/
theorem reference_zeros_not_linear :
    ¬ Execution.WeightedBigO referenceZeros referenceZeroWeights (fun _ => True) id id := by
  rintro ⟨constant, threshold, bounded⟩
  let n := constant + threshold + 1
  obtain ⟨cost, outcome, run, upper⟩ := bounded n trivial (by dsimp [n]; omega)
  have lower := reference_lower (List.replicate n 0) run
  simp only [List.length_replicate] at lower
  change cost ≤ constant*n at upper
  have large : constant < n := by dsimp [n]; omega
  nlinarith

end solvers.«8».complexity
