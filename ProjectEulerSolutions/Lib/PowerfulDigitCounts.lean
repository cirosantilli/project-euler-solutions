import ProjectEulerSolutions.Lib.PowerfulDigits

/-! Finite count recurrences and complete early-stop rules for both source traversal orders. -/
namespace ProjectEulerSolutions.PowerfulDigits

set_option Elab.async false

def sumWindow (weight : Nat → Nat) (first count : Nat) : Nat :=
  ((List.range' first count).map weight).sum

@[simp] theorem sumWindow_zero_length (weight : Nat → Nat) (first : Nat) : sumWindow weight first 0 = 0 := rfl

theorem sumWindow_succ (weight : Nat → Nat) (first count : Nat) :
    sumWindow weight first (count + 1) = weight first + sumWindow weight (first + 1) count := by
  simp only [sumWindow, List.range'_succ, List.map_cons, List.sum_cons]

theorem sumWindow_zero (weight : Nat → Nat) (first count : Nat)
    (zero : ∀ index ∈ List.range' first count, weight index = 0) : sumWindow weight first count = 0 := by
  apply List.sum_eq_zero
  intro value member
  obtain ⟨index, inside, rfl⟩ := List.mem_map.mp member
  exact zero index inside

theorem seed_window_zero (base seed first count : Nat) (valid : 2 ≤ base) (inside : seed < base)
    (positive : 1 ≤ first) (failed : seed ^ first < base ^ (first - 1)) :
    sumWindow (hit base seed) first count = 0 := by
  apply sumWindow_zero
  intro index member
  have interval := List.mem_range'_1.mp member
  have failure := failure_later base seed first (index - first) valid (by omega) positive failed
  rw [Nat.add_sub_cancel' interval.1] at failure
  exact hit_zero base seed index failure

theorem exponent_zero (base exponent : Nat)
    (failed : (base - 1) ^ exponent < base ^ (exponent - 1)) : exponentCount base exponent = 0 := by
  apply List.sum_eq_zero
  intro value member
  obtain ⟨seed, inside, rfl⟩ := List.mem_map.mp member
  have interval := List.mem_range'_1.mp inside
  have bound : seed ≤ base - 1 := by omega
  exact hit_zero base seed exponent ((Nat.pow_le_pow_left bound exponent).trans_lt failed)

theorem exponent_window_zero (base first count : Nat) (valid : 2 ≤ base) (positive : 1 ≤ first)
    (failed : (base - 1) ^ first < base ^ (first - 1)) :
    sumWindow (exponentCount base) first count = 0 := by
  apply sumWindow_zero
  intro index member
  have interval := List.mem_range'_1.mp member
  have failure := failure_later base (base - 1) first (index - first) valid (by omega) positive failed
  rw [Nat.add_sub_cancel' interval.1] at failure
  exact exponent_zero base index failure

theorem window_tail_step (weight : Nat → Nat) (base first : Nat) (before : first < bound base) :
    sumWindow weight first (bound base - first) =
      weight first + sumWindow weight (first + 1) (bound base - (first + 1)) := by
  have length : bound base - first = (bound base - (first + 1)) + 1 := by omega
  rw [length, sumWindow_succ]

theorem seed_window (base seed : Nat) : sumWindow (hit base seed) 1 (bound base - 1) = seedCount base seed := rfl

theorem exponent_window (base exponent : Nat) :
    sumWindow (fun seed => hit base seed exponent) 1 (base - 1) = exponentCount base exponent := rfl

theorem seed_sum (base : Nat) : sumWindow (seedCount base) 1 (base - 1) = answer base := rfl

theorem exponent_sum (base : Nat) : sumWindow (exponentCount base) 1 (bound base - 1) = answer base := transpose base

/-- A successful test always lies strictly before the derived complete bound. -/
theorem successful_before_bound (base seed exponent : Nat) (valid : 2 ≤ base) (inside : seed < base)
    (yes : base ^ (exponent - 1) ≤ seed ^ exponent) : exponent < bound base := by
  by_contra late
  have failure := after_bound base seed exponent valid inside (by omega)
  omega

end ProjectEulerSolutions.PowerfulDigits
