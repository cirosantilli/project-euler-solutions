import ProjectEulerStatements.P64
import Mathlib.Algebra.BigOperators.Group.List.Basic

namespace ProjectEulerSolutions.SqrtPeriodCounts
open ProjectEulerStatements.SqrtReducedStates
set_option Elab.async false

def weight (n : Nat) : Nat := if length n % 2 = 1 then 1 else 0
def count (first size : Nat) : Nat := ((List.range' first size).map weight).sum
def answer (limit : Int) : Nat := count 2 (limit - 1).toNat

@[simp] theorem count_zero (first : Nat) : count first 0 = 0 := rfl
theorem count_succ (first size : Nat) : count first (size + 1) = weight first + count (first + 1) size := by
  simp only [count, List.range'_succ, List.map_cons, List.sum_cons]

@[simp] theorem weight_zero : weight 0 = 0 := by
  simp only [weight, length_of_square 0 rfl]; decide
@[simp] theorem weight_one : weight 1 = 0 := by
  simp only [weight, length_of_square 1 rfl]; decide

theorem skip_two (size : Nat) : count 0 (size + 2) = count 2 size := by
  rw [show size + 2 = (size + 1) + 1 by omega, count_succ, count_succ]
  simp

theorem reference_window (limit : Int) : count 0 (limit + 1).toNat = answer limit := by
  by_cases small : limit ≤ 0
  · have right : (limit - 1).toNat = 0 := by omega
    unfold answer
    rw [right, count_zero]
    by_cases zero : limit = 0
    · subst limit; simp only [show (0 + 1 : Int).toNat = 0 + 1 from rfl, count_succ, count_zero, weight_zero]
    · have left : (limit + 1).toNat = 0 := by omega
      rw [left, count_zero]
  · have shifted : (limit + 1).toNat = (limit - 1).toNat + 2 := by omega
    rw [shifted, skip_two]
    rfl

private theorem fold_count (values : List Nat) (total : Nat) :
    values.foldl (fun acc n => if ProjectEulerStatements.P64.periodLen n % 2 = 1 then acc + 1 else acc) total =
      total + (values.map weight).sum := by
  induction values generalizing total with
  | nil => simp
  | cons head rest ih =>
      by_cases odd : length head % 2 = 1
      · simpa only [List.foldl_cons, ProjectEulerStatements.P64.periodLen, odd, if_true, List.map_cons,
          List.sum_cons, weight, Nat.add_assoc] using ih (total + 1)
      · simpa only [List.foldl_cons, ProjectEulerStatements.P64.periodLen, odd, if_false, List.map_cons,
          List.sum_cons, weight, Nat.zero_add] using ih total

theorem answer_eq_naive (limit : Nat) : answer limit = ProjectEulerStatements.P64.naive limit := by
  rw [← reference_window]
  unfold ProjectEulerStatements.P64.naive ProjectEulerStatements.P64.countOddPeriods
  rw [fold_count]
  simp only [Nat.zero_add, count, List.range_eq_range', Nat.zero_add]
  congr 2

end ProjectEulerSolutions.SqrtPeriodCounts
