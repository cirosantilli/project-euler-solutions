import Mathlib.Data.Nat.Digits.Lemmas
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Nat.Find
import Mathlib.Tactic.Linarith

/-! Complete digit-factorial searches in every numeral base greater than one. -/
namespace ProjectEulerSolutions.DigitFactorialBase
set_option Elab.async false

theorem fold_factorials (digits : List Nat) (total : Nat) :
    digits.foldl (fun acc digit => acc + Nat.factorial digit) total = total + (digits.map Nat.factorial).sum := by
  induction digits generalizing total with
  | nil => simp
  | cons digit rest ih => simp [List.foldl_cons, ih, Nat.add_assoc]

def digitSum (base n : Nat) : Nat := ((Nat.digits base n).map Nat.factorial).sum
def facts (base : Nat) : List Nat := (List.range base).map Nat.factorial
def maxFact (base : Nat) : Nat := Nat.factorial (base - 1)

@[simp] theorem digitSum_zero (base : Nat) : digitSum base 0 = 0 := by simp [digitSum]

theorem digitSum_step (base n : Nat) (valid : 2 ≤ base) (positive : 0 < n) :
    digitSum base n = Nat.factorial (n % base) + digitSum base (n / base) := by
  simp only [digitSum, Nat.digits_def' (by omega : 1 < base) positive, List.map_cons, List.sum_cons]

@[simp] theorem facts_length (base : Nat) : (facts base).length = base := by simp [facts]

theorem facts_get (base digit : Nat) (inside : digit < base) : (facts base)[digit]? = some (Nat.factorial digit) := by
  simp [facts, List.getElem?_range inside]

theorem facts_append (base : Nat) : facts (base + 1) = facts base ++ [Nat.factorial base] := by
  simp [facts, List.range_succ, List.map_append]

private theorem two_pow (n : Nat) : n + 1 ≤ 2 ^ n := by
  induction n with
  | zero => decide
  | succ n ih => rw [Nat.pow_succ]; omega

/-- Exponential place values eventually exceed the maximum possible digit-factorial sum. -/
theorem exists_bad_width (base : Nat) (valid : 2 ≤ base) :
    ∃ digits : Nat, 1 ≤ digits ∧ digits * maxFact base < base ^ (digits - 1) := by
  let maximum := maxFact base
  refine ⟨2 * maximum + 2, by omega, ?_⟩
  have bound := two_pow maximum
  have squared := Nat.mul_le_mul bound bound
  have power : 2 ^ (2 * maximum + 1) = 2 * (2 ^ maximum * 2 ^ maximum) := by
    rw [show 2 * maximum + 1 = maximum + maximum + 1 by omega, Nat.pow_succ, Nat.pow_add]
    omega
  have baseBound := Nat.pow_le_pow_left valid (2 * maximum + 1)
  have same : 2 * maximum + 2 - 1 = 2 * maximum + 1 := by omega
  rw [same]
  rw [power] at baseBound
  nlinarith

noncomputable def stoppingWidth (base : Nat) (valid : 2 ≤ base) : Nat := Nat.find (exists_bad_width base valid)
noncomputable def upper (base : Nat) (valid : 2 ≤ base) : Nat := (stoppingWidth base valid - 1) * maxFact base

theorem stopping_spec (base : Nat) (valid : 2 ≤ base) :
    1 ≤ stoppingWidth base valid ∧ stoppingWidth base valid * maxFact base < base ^ (stoppingWidth base valid - 1) :=
  Nat.find_spec (exists_bad_width base valid)

theorem before_stopping (base : Nat) (valid : 2 ≤ base) (digits : Nat)
    (positive : 1 ≤ digits) (before : digits < stoppingWidth base valid) : base ^ (digits - 1) ≤ digits * maxFact base := by
  have absent := Nat.find_min (exists_bad_width base valid) before
  omega

theorem bad_succ (base digits : Nat) (valid : 2 ≤ base) (positive : 1 ≤ digits)
    (bad : digits * maxFact base < base ^ (digits - 1)) :
    (digits + 1) * maxFact base < base ^ digits := by
  have scale : digits + 1 ≤ base * digits := by nlinarith
  have scaled := Nat.mul_le_mul_right (maxFact base) scale
  have strict := Nat.mul_lt_mul_of_pos_left bad (show 0 < base by omega)
  have exponent : base * base ^ (digits - 1) = base ^ digits := by
    rw [Nat.mul_comm, ← Nat.pow_succ]
    congr 1
    omega
  rw [exponent] at strict
  nlinarith

theorem after_stopping (base : Nat) (valid : 2 ≤ base) (extra : Nat) :
    (stoppingWidth base valid + extra) * maxFact base < base ^ (stoppingWidth base valid + extra - 1) := by
  induction extra with
  | zero => simpa using (stopping_spec base valid).2
  | succ extra ih =>
      have positive := (stopping_spec base valid).1
      simpa only [Nat.add_assoc, Nat.add_sub_cancel] using bad_succ base (stoppingWidth base valid + extra) valid (by omega) ih

/-- At least the two-digit interval is included in the bound. -/
theorem upper_ge_base (base : Nat) (valid : 2 ≤ base) : base ≤ upper base valid := by
  have factorialBound := Nat.self_le_factorial (base - 1)
  have maximum : base ≤ 2 * maxFact base := by unfold maxFact; omega
  have stopped := (stopping_spec base valid).2
  have stopLarge : 3 ≤ stoppingWidth base valid := by
    have positive := (stopping_spec base valid).1
    by_contra small
    have cases : stoppingWidth base valid = 1 ∨ stoppingWidth base valid = 2 := by omega
    rcases cases with same | same
    · rw [same] at stopped
      have pos := Nat.factorial_pos (base - 1)
      simp only [Nat.one_mul, Nat.sub_self, pow_zero] at stopped
      omega
    · have bad : 2 * maxFact base < base := by
        simpa only [same, show (2 : Nat) - 1 = 1 from rfl, pow_one] using stopped
      omega
  unfold upper
  have monotone := Nat.mul_le_mul_right (maxFact base) (show 2 ≤ stoppingWidth base valid - 1 by omega)
  omega

private theorem sum_factorials_le (base : Nat) (digits : List Nat) (bounded : ∀ digit ∈ digits, digit < base) :
    (digits.map Nat.factorial).sum ≤ digits.length * maxFact base := by
  induction digits with
  | nil => simp
  | cons digit rest ih =>
      have small := bounded digit (by simp)
      have factorialBound := Nat.factorial_le (show digit ≤ base - 1 by omega)
      have tail := ih (by intro value member; exact bounded value (by simp [member]))
      simp only [List.map_cons, List.sum_cons, List.length_cons]
      unfold maxFact at *
      nlinarith

/-- Every positive fixed point lies inside the derived bound, in every supported base. -/
theorem fixed_bounded (base n : Nat) (valid : 2 ≤ base) (positive : 0 < n) (fixed : digitSum base n = n) :
    n ≤ upper base valid := by
  let digits := (Nat.digits base n).length
  have digitPositive : 0 < digits := List.length_pos_iff.mpr (Nat.digits_ne_nil_iff_ne_zero.mpr (by omega))
  have lower : base ^ (digits - 1) ≤ n :=
    (Nat.lt_digits_length_iff (by omega : 1 < base) n).mp (by change digits - 1 < digits; omega)
  have sumBound : n ≤ digits * maxFact base := by
    rw [← fixed]
    exact sum_factorials_le base (Nat.digits base n) (by intro digit member; exact Nat.digits_lt_base (by omega) member)
  have small : digits < stoppingWidth base valid := by
    by_contra late
    have tooLarge := after_stopping base valid (digits - stoppingWidth base valid)
    have same : stoppingWidth base valid + (digits - stoppingWidth base valid) = digits := by omega
    rw [same] at tooLarge
    omega
  exact sumBound.trans (Nat.mul_le_mul_right _ (show digits ≤ stoppingWidth base valid - 1 by omega))

def foldValues (base : Nat) (items : List Nat) (total : Nat) : Nat :=
  items.foldl (fun acc n => if digitSum base n = n then acc + n else acc) total

noncomputable def solutions (base : Nat) (valid : 2 ≤ base) : List Nat :=
  (List.range' base (upper base valid + 1 - base)).filter (fun n => decide (digitSum base n = n))
noncomputable def answer (base : Nat) (valid : 2 ≤ base) : Nat := (solutions base valid).sum

theorem mem_solutions (base n : Nat) (valid : 2 ≤ base) :
    n ∈ solutions base valid ↔ base ≤ n ∧ digitSum base n = n := by
  simp only [solutions, List.mem_filter, decide_eq_true_eq, List.mem_range']
  constructor
  · rintro ⟨⟨offset, bound, same⟩, fixed⟩
    exact ⟨by omega, fixed⟩
  · rintro ⟨lower, fixed⟩
    have upper := fixed_bounded base n valid (by omega) fixed
    exact ⟨⟨n - base, by omega, by omega⟩, fixed⟩

theorem fold_sum (base : Nat) (items : List Nat) (total : Nat) :
    foldValues base items total = total + (items.filter (fun n => decide (digitSum base n = n))).sum := by
  unfold foldValues
  induction items generalizing total with
  | nil => simp
  | cons first rest ih =>
      by_cases fixed : digitSum base first = first <;>
        simp [List.foldl_cons, fixed, ih, Nat.add_assoc]

theorem range_answer (base : Nat) (valid : 2 ≤ base) :
    foldValues base (List.range' base (upper base valid + 1 - base)) 0 = answer base valid := by
  simp only [fold_sum, Nat.zero_add, answer, solutions]

end ProjectEulerSolutions.DigitFactorialBase
