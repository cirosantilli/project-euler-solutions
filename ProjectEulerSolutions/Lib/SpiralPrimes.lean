import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Linarith

/-! Finite spiral-prefix mathematics, independent of the old P58 search and termination files. -/
namespace ProjectEulerSolutions.SpiralPrimes
set_option Elab.async false

variable {num den bound left right layer start fuel : Nat}

def side (layer : Nat) : Nat := 2 * layer + 1
def square (layer : Nat) : Nat := (side layer) ^ 2
def step (layer : Nat) : Nat := 2 * layer
def corner (layer offset : Nat) : Nat := square layer - offset * step layer

def corners (layer : Nat) : List Nat := [corner layer 1, corner layer 2, corner layer 3]
def allCorners (layer : Nat) : List Nat := square layer :: corners layer

def primeCount (values : List Nat) : Nat := (values.filter fun value => decide (Nat.Prime value)).length

def row (layer : Nat) : Nat := primeCount (corners layer)

def count : Nat → Nat
  | 0 => 0
  | layer + 1 => count layer + row (layer + 1)

def countWithSquare : Nat → Nat
  | 0 => 0
  | layer + 1 => countWithSquare layer + primeCount (allCorners (layer + 1))

def total (layer : Nat) : Nat := 4 * layer + 1

def below (num den layer : Nat) : Prop := den * count layer < num * total layer

instance (num den layer : Nat) : Decidable (below num den layer) := inferInstanceAs (Decidable (_ < _))

@[simp] theorem count_zero : count 0 = 0 := rfl
@[simp] theorem count_succ (layer : Nat) : count (layer + 1) = count layer + row (layer + 1) := rfl

theorem square_not_prime (layer : Nat) : ¬ Nat.Prime (square layer) :=
  Nat.Prime.not_prime_pow (by decide)

theorem allCorners_primeCount (layer : Nat) : primeCount (allCorners layer) = row layer := by
  have composite := square_not_prime layer
  simp [allCorners, primeCount, row, composite]

theorem countWithSquare_eq (layer : Nat) : countWithSquare layer = count layer := by
  induction layer with
  | zero => rfl
  | succ layer ih =>
      simp only [countWithSquare, count_succ, ih, allCorners_primeCount]

theorem total_eq_reference (layer : Nat) : 2 * side layer - 1 = total layer := by
  simp only [side, total]
  omega

theorem corner_offset_lt_square (layer offset : Nat) (offsetBound : offset ≤ 3) :
    offset * step layer < square layer := by
  by_cases zero : layer = 0
  · subst layer
    simp [step, square, side]
  · have positive : 1 ≤ layer := by omega
    have squareBound : layer ≤ layer * layer := by
      simpa using Nat.mul_le_mul_left layer positive
    interval_cases offset <;> simp only [step, square, side] <;> nlinarith

theorem corner_positive (layer offset : Nat) (offsetBound : offset ≤ 3) : 0 < corner layer offset :=
  Nat.sub_pos_of_lt (corner_offset_lt_square layer offset offsetBound)

theorem corner_at_least_three (layer offset : Nat) (positive : 0 < layer) (offsetBound : offset ≤ 3) :
    3 ≤ corner layer offset := by
  apply Nat.le_sub_of_add_le
  have squareBound : layer ≤ layer * layer := by
    simpa using Nat.mul_le_mul_left layer (show 1 ≤ layer by omega)
  interval_cases offset <;> simp only [step, square, side] <;> nlinarith

/-- The exact finite condition needed to justify both source loops; no existence axiom is used. -/
structure FirstCrossing (num den bound : Nat) : Prop where
  positive : 0 < bound
  crossing : below num den bound
  earlier : ∀ layer, 0 < layer → layer < bound → ¬ below num den layer

theorem FirstCrossing.unique (first : FirstCrossing num den left) (second : FirstCrossing num den right) :
    left = right := by
  rcases lt_trichotomy left right with smaller | same | larger
  · exact False.elim (second.earlier left first.positive smaller first.crossing)
  · exact same
  · exact False.elim (first.earlier right second.positive larger second.crossing)

/-- A bounded view of the reference's increasing scan, starting at a positive layer. -/
def search (num den : Nat) : Nat → Nat → Option Nat
  | 0, _ => none
  | fuel + 1, layer =>
      if below num den layer then some (side layer) else search num den fuel (layer + 1)

/-- A bounded view of the root's count-update-before-test scan. -/
def rootSearch (num den : Nat) : Nat → Nat → Nat → Option Nat
  | 0, _, _ => none
  | fuel + 1, previous, primes =>
      let layer := previous + 1
      let updated := primes + row layer
      if den * updated < num * total layer then some (side layer)
      else rootSearch num den fuel layer updated

theorem rootSearch_eq_search (num den fuel previous : Nat) :
    rootSearch num den fuel previous (count previous) = search num den fuel (previous + 1) := by
  induction fuel generalizing previous with
  | zero => rfl
  | succ fuel ih =>
      change (if below num den (previous + 1) then some (side (previous + 1))
        else rootSearch num den fuel (previous + 1) (count (previous + 1))) =
        (if below num den (previous + 1) then some (side (previous + 1))
          else search num den fuel ((previous + 1) + 1))
      split
      · rfl
      · exact ih (previous + 1)

theorem search_of_firstCrossing (first : FirstCrossing num den bound)
    (positive : 0 < start) (inside : start ≤ bound) (window : start + fuel = bound + 1) :
    search num den fuel start = some (side bound) := by
  induction fuel generalizing start with
  | zero => omega
  | succ fuel ih =>
      by_cases last : start = bound
      · subst start
        rw [search, if_pos first.crossing]
      · have earlier : start < bound := by omega
        rw [search, if_neg (first.earlier start positive earlier)]
        exact ih (start := start + 1) (by omega) (by omega) (by omega)

theorem FirstCrossing.search (first : FirstCrossing num den bound) :
    search num den bound 1 = some (side bound) :=
  search_of_firstCrossing first (by decide) (by have := first.positive; omega) (by omega)

theorem FirstCrossing.rootSearch (first : FirstCrossing num den bound) :
    rootSearch num den bound 0 0 = some (side bound) := by
  exact (rootSearch_eq_search num den bound 0).trans first.search

/-- A row certificate may be generated externally, but each classification is a checked proof. -/
structure RowCertificate (layer : Nat) where
  first : Bool
  second : Bool
  third : Bool
  first_correct : decide (Nat.Prime (corner layer 1)) = first
  second_correct : decide (Nat.Prime (corner layer 2)) = second
  third_correct : decide (Nat.Prime (corner layer 3)) = third

def RowCertificate.increment (certificate : RowCertificate layer) : Nat :=
  (if certificate.first then 1 else 0) + (if certificate.second then 1 else 0) +
    (if certificate.third then 1 else 0)

theorem RowCertificate.checked (certificate : RowCertificate layer) : row layer = certificate.increment := by
  obtain ⟨first, second, third, firstCorrect, secondCorrect, thirdCorrect⟩ := certificate
  cases first <;> cases second <;> cases third <;>
    simp [row, primeCount, corners, firstCorrect, secondCorrect, thirdCorrect, RowCertificate.increment]

/-- Finite checked cumulative counters; values outside the certified prefix are irrelevant. -/
structure CountPrefix (bound : Nat) where
  values : Nat → Nat
  initial : values 0 = 0
  next : ∀ layer, layer < bound → values (layer + 1) = values layer + row (layer + 1)

theorem CountPrefix.correct (certificate : CountPrefix bound) (inside : layer ≤ bound) :
    certificate.values layer = count layer := by
  induction layer with
  | zero => exact certificate.initial
  | succ layer ih =>
      rw [certificate.next layer (by omega), ih (by omega), count_succ]

theorem CountPrefix.firstCrossing (certificate : CountPrefix bound) (positive : 0 < bound)
    (last : den * certificate.values bound < num * total bound)
    (earlier : ∀ layer, 0 < layer → layer < bound → num * total layer ≤ den * certificate.values layer) :
    FirstCrossing num den bound := by
  refine ⟨positive, ?_, ?_⟩
  · simpa only [below, certificate.correct (Nat.le_refl bound)] using last
  · intro layer positiveLayer before
    have checked := earlier layer positiveLayer before
    rw [certificate.correct (by omega)] at checked
    exact Nat.not_lt_of_ge checked

end ProjectEulerSolutions.SpiralPrimes
