import lang.py.FormalSW.Permutations
import ProjectEulerStatements.Lib.ListMaximum
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.List.Perm.Basic
import Mathlib.Data.Nat.Digits.Lemmas
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Permutation maxima and the pruning used by the pandigital-prime search. -/
namespace ProjectEulerSolutions.PandigitalPrimeSearch
open FormalSW.PyAst

set_option Elab.async false

def pool (width : Nat) : List Nat := List.range' 1 width
def rows (width : Nat) : List (List Nat) := Permutations.rows width (pool width)
def value (digits : List Nat) : Nat := Nat.ofDigits 10 digits.reverse
def lastDigit (digits : List Nat) : Nat := digits.reverse.headD 0
def allowed (digits : List Nat) : Bool := !(lastDigit digits ∈ [2, 4, 5, 6, 8])
def candidates (width : Nat) : List Nat := ((rows width).map value).filter (fun n => decide (Nat.Prime n))
def best (width : Nat) : Nat := (candidates width).foldl max 0
def through (width : Nat) : Nat := (List.range' 1 width).foldl (fun acc n => max acc (best n)) 0

theorem fold_value (digits : List Nat) (initial : Nat) :
    digits.foldl (fun acc d => acc*10+d) initial = initial * 10^digits.length + value digits := by
  induction digits generalizing initial with
  | nil => simp [value]
  | cons head tail ih =>
      rw [List.foldl_cons, ih]
      simp only [List.length_cons, value, Nat.ofDigits_reverse_cons, Nat.pow_succ]
      ring

theorem row_perm (width : Nat) (digits : List Nat) (member : digits ∈ rows width) :
    digits.Perm (pool width) :=
  Permutations.row_perm width (pool width) digits (by simp [pool]) member

theorem row_length (width : Nat) (digits : List Nat) (member : digits ∈ rows width) :
    digits.length = width := by
  simpa only [pool, List.length_range'] using (row_perm width digits member).length_eq

theorem row_digits (width : Nat) (digits : List Nat) (member : digits ∈ rows width) :
    ∀ d ∈ digits, 1 ≤ d ∧ d ≤ width := by
  intro d inside
  have found := (row_perm width digits member).mem_iff.mp inside
  have bounded := List.mem_range'.mp found
  omega

/-- The lower bound depends only on the nonzero first decimal digit. -/
theorem value_lower (digits : List Nat) (nonempty : digits ≠ [])
    (positive : ∀ d ∈ digits, 1 ≤ d) : 10 ^ (digits.length - 1) ≤ value digits := by
  cases digits with
  | nil => contradiction
  | cons head tail =>
      have lower := positive head (by simp)
      simp only [value, Nat.ofDigits_reverse_cons, List.length_cons, Nat.add_sub_cancel]
      have product := Nat.mul_le_mul_left (10 ^ tail.length) lower
      omega

theorem value_upper (digits : List Nat) (decimal : ∀ d ∈ digits, d < 10) :
    value digits < 10 ^ digits.length := by
  simpa only [value, List.length_reverse] using
    Nat.ofDigits_lt_base_pow_length (b := 10) (l := digits.reverse) (by decide)
      (fun d member => decimal d (List.mem_reverse.mp member))

theorem row_bounds (width : Nat) (positive : 0 < width) (bounded : width ≤ 9)
    (digits : List Nat) (member : digits ∈ rows width) :
    10 ^ (width - 1) ≤ value digits ∧ value digits < 10 ^ width := by
  have length := row_length width digits member
  have numbers := row_digits width digits member
  exact ⟨by simpa only [length] using
    value_lower digits (by intro empty; simp [empty] at length; omega) (fun d h => (numbers d h).1),
    by simpa only [length] using value_upper digits (fun d h => by have := numbers d h; omega)⟩

private theorem pool_sum (width : Nat) (bounded : width ≤ 9) :
    (pool width).sum = width * (width + 1) / 2 := by
  interval_cases width <;> decide +kernel

theorem value_mod_three (width : Nat) (bounded : width ≤ 9) (digits : List Nat)
    (member : digits ∈ rows width) :
    value digits % 3 = (width * (width + 1) / 2) % 3 := by
  rw [value, Nat.ofDigits_mod, show 10 % 3 = 1 by decide, Nat.ofDigits_one,
    List.sum_reverse, (row_perm width digits member).sum_eq, pool_sum width bounded]

theorem skip_width (width : Nat) (positive : 0 < width) (bounded : width ≤ 9)
    (skipped : (width * (width + 1) / 2) % 3 = 0) : candidates width = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro n member
  obtain ⟨fromRows, prime⟩ := List.mem_filter.mp member
  obtain ⟨digits, row, rfl⟩ := List.mem_map.mp fromRows
  have lowerWidth : 2 ≤ width := by
    by_contra small
    have same : width = 1 := by omega
    subst width
    contradiction
  have lower := (row_bounds width positive bounded digits row).1
  have atLeastTen : 10 ≤ value digits := le_trans
    (by simpa using Nat.pow_le_pow_right (by decide : 1 ≤ 10) (by omega : 1 ≤ width-1)) lower
  have divides : 3 ∣ value digits := Nat.dvd_of_mod_eq_zero
    ((value_mod_three width bounded digits row).trans skipped)
  rcases (Nat.dvd_prime (of_decide_eq_true prime)).mp divides with one | same <;> omega

theorem value_mod_last (digits : List Nat) (modulus : Nat) (divides : modulus ∣ 10) :
    value digits % modulus = lastDigit digits % modulus := by
  rw [value, Nat.ofDigits_mod, Nat.mod_eq_zero_of_dvd divides]
  simp only [lastDigit]
  cases digits.reverse <;> simp [Nat.ofDigits]

/-- A prime candidate cannot have one of the filtered final decimal digits. -/
theorem prime_allowed (width : Nat) (positive : 0 < width) (bounded : width ≤ 9)
    (digits : List Nat) (member : digits ∈ rows width) (prime : Nat.Prime (value digits)) :
    allowed digits = true := by
  by_cases single : width = 1
  · subst width
    have same : digits = [1] := by
      have permuted := row_perm 1 digits member
      simpa only [pool, List.range'_succ, List.range'_zero, List.perm_singleton] using permuted
    subst digits
    decide
  have lower := (row_bounds width positive bounded digits member).1
  have atLeastTen : 10 ≤ value digits := le_trans
    (by simpa using Nat.pow_le_pow_right (by decide : 1 ≤ 10) (by omega : 1 ≤ width-1)) lower
  have nonzero (p : Nat) (pPrime : Nat.Prime p) (small : p < 10) : value digits % p ≠ 0 := by
    intro zero
    rcases (Nat.dvd_prime prime).mp (Nat.dvd_of_mod_eq_zero zero) with one | same
    · have := pPrime.two_le; omega
    · omega
  have even := nonzero 2 Nat.prime_two (by decide)
  have five := nonzero 5 (by decide) (by decide)
  rw [value_mod_last digits 2 (by decide)] at even
  rw [value_mod_last digits 5 (by decide)] at five
  change (!decide (lastDigit digits ∈ [2, 4, 5, 6, 8])) = true
  simp only [Bool.not_eq_true', decide_eq_false_iff_not]
  intro forbidden
  simp only [List.mem_cons, List.not_mem_nil, or_false] at forbidden
  rcases forbidden with same | same | same | same | same <;> omega

def select (current : Nat) (digits : List Nat) : Nat :=
  if allowed digits && decide (current < value digits ∧ Nat.Prime (value digits)) then
    value digits
  else current

theorem select_correct (digits : List Nat) (current : Nat)
    (safe : Nat.Prime (value digits) → allowed digits = true) :
    select current digits = if Nat.Prime (value digits) then max current (value digits) else current := by
  by_cases prime : Nat.Prime (value digits)
  · have yes := safe prime
    by_cases better : current < value digits
    · simp [select, yes, prime, better, max_eq_right (by omega : current ≤ value digits)]
    · simp [select, yes, prime, better, max_eq_left (by omega : value digits ≤ current)]
  · simp [select, prime]

theorem scan_correct (items : List (List Nat)) (current : Nat)
    (safe : ∀ digits ∈ items, Nat.Prime (value digits) → allowed digits = true) :
    items.foldl select current =
      ((items.map value).filter (fun n => decide (Nat.Prime n))).foldl max current := by
  induction items generalizing current with
  | nil => rfl
  | cons head tail ih =>
      rw [List.foldl_cons, select_correct head current (safe head (by simp))]
      have rest : ∀ digits ∈ tail, Nat.Prime (value digits) → allowed digits = true :=
        fun digits member => safe digits (by simp [member])
      by_cases prime : Nat.Prime (value head)
      · simpa only [prime, if_true, List.map_cons, List.filter_cons, decide_true] using
          ih (max current (value head)) rest
      · simpa only [prime, if_false, List.map_cons, List.filter_cons, decide_false] using
          ih current rest

def optimizedWidth (width : Nat) : Nat :=
  if (width * (width + 1) / 2) % 3 = 0 then 0 else (rows width).foldl select 0

theorem width_correct (width : Nat) (positive : 0 < width) (bounded : width ≤ 9) :
    optimizedWidth width = best width := by
  unfold optimizedWidth
  by_cases skipped : (width * (width + 1) / 2) % 3 = 0
  · simp only [if_pos skipped, best, skip_width width positive bounded skipped, List.foldl_nil]
  · rw [if_neg skipped]
    exact scan_correct (rows width) 0 (prime_allowed width positive bounded)

@[simp] theorem through_zero : through 0 = 0 := rfl

theorem through_succ (width : Nat) :
    through (width+1) = max (through width) (best (width+1)) := by
  simp only [through, List.range'_1_concat, List.foldl_append, List.foldl_cons,
    List.foldl_nil, Nat.add_comm 1 width]

theorem best_upper (width : Nat) (bounded : width ≤ 9) : best width < 10 ^ width := by
  have positive : 0 < 10 ^ width := Nat.pow_pos (by decide)
  have bound : best width ≤ 10 ^ width - 1 := by
    apply (ProjectEulerStatements.ListMaximum.fold_le_iff (candidates width) 0 _).mpr
    refine ⟨by omega, ?_⟩
    intro n member
    obtain ⟨fromRows, _⟩ := List.mem_filter.mp member
    obtain ⟨digits, row, rfl⟩ := List.mem_map.mp fromRows
    have below := value_upper digits (fun d inside => by
      have := row_digits width digits row d inside
      omega)
    rw [row_length width digits row] at below
    omega
  omega

theorem best_lower (width : Nat) (positive : 0 < width) (bounded : width ≤ 9)
    (nonzero : best width ≠ 0) : 10 ^ (width-1) ≤ best width := by
  cases values : candidates width with
  | nil => simp [best, values] at nonzero
  | cons first rest =>
      have member : first ∈ candidates width := by rw [values]; simp
      obtain ⟨fromRows, _⟩ := List.mem_filter.mp member
      obtain ⟨digits, row, equal⟩ := List.mem_map.mp fromRows
      have lower := (row_bounds width positive bounded digits row).1
      have included := ProjectEulerStatements.ListMaximum.le_fold_of_mem (candidates width) 0 first member
      rw [equal] at lower
      exact lower.trans included

theorem through_upper (width : Nat) (bounded : width ≤ 9) : through width < 10 ^ width := by
  induction width with
  | zero => decide
  | succ width ih =>
      rw [through_succ, max_lt_iff]
      refine ⟨lt_of_lt_of_le (ih (by omega)) ?_, best_upper (width+1) bounded⟩
      exact Nat.pow_le_pow_right (by decide) (by omega)

/-- This is the descending search, including its first nonzero early return. -/
def search : Nat → Nat
  | 0 => 0
  | width+1 =>
      let found := optimizedWidth (width+1)
      if found = 0 then search width else found

/-- Any positive prime from a wider block dominates all candidates in narrower blocks. -/
theorem search_correct (width : Nat) (bounded : width ≤ 9) : search width = through width := by
  induction width with
  | zero => rfl
  | succ width ih =>
      rw [search, width_correct (width+1) (by omega) bounded, through_succ]
      by_cases empty : best (width+1) = 0
      · simpa only [empty, if_true, max_zero] using ih (by omega)
      · rw [if_neg empty]
        have lower := best_lower (width+1) (by omega) bounded empty
        have upper := through_upper width (by omega)
        have ordered : through width ≤ best (width+1) := by
          rw [Nat.add_sub_cancel] at lower
          omega
        exact (max_eq_right ordered).symm

end ProjectEulerSolutions.PandigitalPrimeSearch
