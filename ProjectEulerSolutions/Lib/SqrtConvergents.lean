import ProjectEulerStatements.P57
import ProjectEulerStatements.Lib.DecimalDigits
import Init.Data.List.Range
import Lean.Elab.Tactic.Omega

/-! Positive sqrt-two convergents and structurally finite counting, without old root P57 dependencies. -/
namespace ProjectEulerSolutions.SqrtConvergents
set_option Elab.async false

abbrev convergent := ProjectEulerStatements.P57.convergent

def indicator (p q : Nat) : Nat :=
  if (Nat.digits 10 q).length < (Nat.digits 10 p).length then 1 else 0

def weight (index : Nat) : Nat := indicator (convergent index).1 (convergent index).2

def sumFrom (first length : Nat) : Nat := ((List.range' first length).map weight).sum

def loop : Nat → Nat → Nat → Nat → Nat
  | 0, _, _, total => total
  | remaining+1, p, q, total =>
      loop remaining (p+2*q) (p+q) (total+indicator p q)

theorem convergent_zero : convergent 0 = (3, 2) := rfl

theorem convergent_next (index : Nat) :
    convergent (index+1) = ((convergent index).1 + 2*(convergent index).2,
      (convergent index).1 + (convergent index).2) := rfl

/-- Positivity excludes the discrepancy between the empty positional digits of zero and the string "0". -/
theorem convergent_positive (index : Nat) : 0 < (convergent index).1 ∧ 0 < (convergent index).2 := by
  induction index with
  | zero => decide
  | succ index ih =>
      change 0 < (convergent index).1 + 2*(convergent index).2 ∧
        0 < (convergent index).1 + (convergent index).2
      omega

/-- The actual string length counts exactly the positional digits of each positive numerator/denominator. -/
theorem printed_length (n : Nat) (positive : 0 < n) :
    (toString (n : Int)).length = (Nat.digits 10 n).length := by
  simpa only [String.length_toList] using ProjectEulerStatements.DecimalDigits.int_string_length n positive

theorem string_indicator_eq (p q : Nat) (positiveP : 0 < p) (positiveQ : 0 < q) :
    (if (toString (q : Int)).length < (toString (p : Int)).length then 1 else 0) = indicator p q := by
  rw [printed_length p positiveP, printed_length q positiveQ]
  rfl

theorem sumFrom_zero (first : Nat) : sumFrom first 0 = 0 := rfl

theorem sumFrom_succ (first length : Nat) :
    sumFrom first (length+1) = weight first + sumFrom (first+1) length := by
  simp only [sumFrom, List.range'_succ, List.map_cons, List.sum_cons]

private theorem fold_count (indices : List Nat) (initial : Nat) :
    indices.foldl (fun total index =>
      if (Nat.digits 10 (convergent index).2).length < (Nat.digits 10 (convergent index).1).length
      then total+1 else total) initial = initial + (indices.map weight).sum := by
  induction indices generalizing initial with
  | nil => simp
  | cons index rest ih =>
      by_cases more : (Nat.digits 10 (convergent index).2).length < (Nat.digits 10 (convergent index).1).length
      · have first : weight index = 1 := by simp [weight, indicator, more]
        rw [List.foldl_cons, if_pos more, ih, List.map_cons, List.sum_cons, first]
        omega
      · have first : weight index = 0 := by simp [weight, indicator, more]
        rw [List.foldl_cons, if_neg more, ih, List.map_cons, List.sum_cons, first]
        omega

/-- The genuine structural reference model is the finite sum over the same zero-based convergents. -/
theorem naive_eq_sumFrom (limit : Nat) : ProjectEulerStatements.P57.naive limit = sumFrom 0 limit := by
  unfold ProjectEulerStatements.P57.naive
  simpa only [sumFrom, List.range_eq_range', Nat.zero_add] using fold_count (List.range limit) 0

/-- The iterative recurrence and recursive reference traverse precisely the same finite suffix. -/
theorem loop_eq_sumFrom (remaining first initial : Nat) :
    loop remaining (convergent first).1 (convergent first).2 initial = initial + sumFrom first remaining := by
  induction remaining generalizing first initial with
  | zero => simp only [loop, sumFrom_zero, Nat.add_zero]
  | succ remaining ih =>
      rw [loop]
      change loop remaining (convergent (first+1)).1 (convergent (first+1)).2 (initial+weight first) = _
      rw [ih, sumFrom_succ]
      omega

theorem loop_initial_eq_naive (limit : Nat) : loop limit 3 2 0 = ProjectEulerStatements.P57.naive limit := by
  calc
    loop limit 3 2 0 = 0 + sumFrom 0 limit := loop_eq_sumFrom limit 0 0
    _ = ProjectEulerStatements.P57.naive limit := by rw [Nat.zero_add, ← naive_eq_sumFrom]

end ProjectEulerSolutions.SqrtConvergents
