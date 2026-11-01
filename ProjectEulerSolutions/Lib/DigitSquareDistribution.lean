import ProjectEulerSolutions.Lib.DigitSquarePrefixes
import Mathlib.Algebra.BigOperators.Ring.List

/-! Weighted prefix states retain exact multiplicities when several numbers have the same digit-square sum. -/
namespace ProjectEulerSolutions.DigitSquareChains.Prefix
open scoped BigOperators
set_option Elab.async false

abbrev Table := List (State × Nat)

def weighted (table : Table) (score : State → Nat) : Nat :=
  (table.map (fun entry => entry.2 * score entry.1)).sum

def row (ceiling : Nat) (entry : State × Nat) : Table :=
  (List.range (digits ceiling entry.1.2)).map (fun digit => (advance ceiling entry.1 digit, entry.2))

def expand (ceiling : Nat) (table : Table) : Table := table.flatMap (row ceiling)

/-- Equality against every score function states exact equality of the represented finite distributions. -/
def Exact (bound : Nat) (table : Table) : Prop :=
  ∀ score, weighted table score = ∑ value ∈ Finset.range (bound + 1), score (state bound value)

private theorem range_sum (length : Nat) (f : Nat → Nat) :
    ((List.range length).map f).sum = ∑ n ∈ Finset.range length, f n := by
  induction length with
  | zero => simp
  | succ length ih => simp [List.range_succ, Finset.sum_range_succ, ih]

theorem weighted_append (left right : Table) (score : State → Nat) :
    weighted (left ++ right) score = weighted left score + weighted right score := by
  simp [weighted]

theorem weighted_row (ceiling : Nat) (entry : State × Nat) (score : State → Nat) :
    weighted (row ceiling entry) score =
      entry.2 * ∑ digit ∈ Finset.range (digits ceiling entry.1.2), score (advance ceiling entry.1 digit) := by
  simp only [weighted, row, List.map_map, Function.comp_def]
  rw [List.sum_map_mul_left, range_sum]

theorem weighted_expand (ceiling : Nat) (table : Table) (score : State → Nat) :
    weighted (expand ceiling table) score = weighted table
      (fun current => ∑ digit ∈ Finset.range (digits ceiling current.2), score (advance ceiling current digit)) := by
  induction table with
  | nil => simp [expand, weighted]
  | cons head rest ih =>
      change weighted (row ceiling head ++ expand ceiling rest) score = _
      rw [weighted_append, weighted_row, ih]
      rfl

theorem initial_exact : Exact 0 [((0, true), 1)] := by
  intro score
  simp [weighted, state]

theorem branch_size (bound value ceiling : Nat) :
    digits ceiling (state bound value).2 = if value = bound then ceiling + 1 else 10 := by
  by_cases tight : value = bound <;> simp [digits, state, tight]

/-- Expanding all permitted digits preserves the exact distribution at the extended bound. -/
theorem Exact.expand {bound : Nat} {table : Table} (exact : Exact bound table)
    (ceiling : Nat) (small : ceiling < 10) : Exact (10 * bound + ceiling) (expand ceiling table) := by
  intro score
  rw [weighted_expand, exact]
  calc
    (∑ value ∈ Finset.range (bound + 1),
        ∑ digit ∈ Finset.range (digits ceiling (state bound value).2),
          score (advance ceiling (state bound value) digit)) =
        ∑ value ∈ Finset.range (bound + 1),
          ∑ digit ∈ Finset.range (if value = bound then ceiling + 1 else 10),
            score (state (10 * bound + ceiling) (10 * value + digit)) := by
      apply Finset.sum_congr rfl
      intro value member
      rw [branch_size]
      apply Finset.sum_congr rfl
      intro digit included
      have digitBound : digit < 10 := by
        rw [Finset.mem_range] at included
        by_cases tight : value = bound <;> simp only [tight, ↓reduceIte] at included <;> omega
      rw [advance_state bound value ceiling digit small digitBound]
    _ = _ := extend_sum bound ceiling small (fun value => score (state (10 * bound + ceiling) value))

/-- Merging duplicate states is sound whenever it preserves the sum of their multiplicities. -/
theorem Exact.congr {bound : Nat} {left right : Table} (exact : Exact bound left)
    (same : ∀ score, weighted left score = weighted right score) : Exact bound right := by
  intro score
  rw [← same, exact]

/-- Increment the existing state's multiplicity, or insert its first contribution. -/
def add : Table → State → Nat → Table
  | [], key, ways => [(key, ways)]
  | (stored, count) :: rest, key, ways =>
      if stored = key then (stored, count + ways) :: rest else (stored, count) :: add rest key ways

theorem weighted_add (table : Table) (key : State) (ways : Nat) (score : State → Nat) :
    weighted (add table key ways) score = weighted table score + ways * score key := by
  induction table with
  | nil => simp [add, weighted]
  | cons entry rest ih =>
      rcases entry with ⟨stored, count⟩
      by_cases same : stored = key
      · subst stored
        simp [add, weighted, Nat.add_mul, Nat.add_assoc, Nat.add_comm]
      · simp only [add, if_neg same]
        change count * score stored + weighted (add rest key ways) score = _
        rw [ih]
        simp [weighted, Nat.add_assoc]

def collect (initial contributions : Table) : Table :=
  contributions.foldl (fun table entry => add table entry.1 entry.2) initial

theorem collect_append (initial left right : Table) :
    collect initial (left ++ right) = collect (collect initial left) right := by
  simp [collect, List.foldl_append]

theorem weighted_collect (initial contributions : Table) (score : State → Nat) :
    weighted (collect initial contributions) score = weighted initial score + weighted contributions score := by
  induction contributions generalizing initial with
  | nil => simp [collect, weighted]
  | cons entry rest ih =>
      change weighted (collect (add initial entry.1 entry.2) rest) score = _
      rw [ih, weighted_add]
      simp [weighted, Nat.add_assoc]

def transition (ceiling : Nat) (table : Table) : Table := collect [] (expand ceiling table)

theorem Exact.transition {bound : Nat} {table : Table} (exact : Exact bound table)
    (ceiling : Nat) (small : ceiling < 10) : Exact (10 * bound + ceiling) (transition ceiling table) := by
  apply (exact.expand ceiling small).congr
  intro score
  rw [Prefix.transition, weighted_collect]
  simp [weighted]

/-- Any number of decimal digits can be processed; leading zeroes are included in the initial state. -/
theorem Exact.fold_digits {bound : Nat} {table : Table} (exact : Exact bound table)
    (values : List Nat) (valid : ∀ digit ∈ values, digit < 10) :
    Exact (values.foldl (fun bound digit => 10 * bound + digit) bound)
      (values.foldl (fun table digit => Prefix.transition digit table) table) := by
  induction values generalizing bound table with
  | nil => exact exact
  | cons digit rest ih =>
      exact ih (exact.transition digit (valid digit (by simp)))
        (by intro next member; exact valid next (by simp [member]))

def characterValue (character : Char) : Nat := character.toNat - '0'.toNat

theorem characterValue_small (character : Char) (valid : character.isDigit) :
    characterValue character < 10 := by
  have bounds := Char.isDigit_iff_toNat.mp valid
  simp only [show '0'.toNat = 48 from rfl, show '9'.toNat = 57 from rfl] at bounds
  simp only [characterValue, show '0'.toNat = 48 from rfl]
  omega

/-- Decimal character traversal has the same positional value and distribution as digit traversal. -/
theorem Exact.fold_characters {bound : Nat} {table : Table} (exact : Exact bound table)
    (characters : List Char) (valid : ∀ character ∈ characters, character.isDigit) :
    Exact (Nat.ofDigitChars 10 characters bound)
      (characters.foldl (fun table character => Prefix.transition (characterValue character) table) table) := by
  have small : ∀ digit ∈ characters.map characterValue, digit < 10 := by
    intro digit member
    obtain ⟨character, present, rfl⟩ := List.mem_map.mp member
    exact characterValue_small character (valid character present)
  simpa only [List.foldl_map, characterValue, Nat.ofDigitChars] using
    exact.fold_digits (characters.map characterValue) small

private theorem filter_sum (values : List Nat) :
    (values.filter endsAt89).length = (values.map (fun n => if endsAt89 n then 1 else 0)).sum := by
  induction values with
  | nil => rfl
  | cons head rest ih =>
      cases hit : endsAt89 head <;> simp [hit, ih, Nat.add_comm]

theorem count_succ (bound : Nat) :
    count (bound + 1 : Nat) = ∑ n ∈ Finset.range (bound + 1), if endsAt89 n then 1 else 0 := by
  have length : ((bound + 1 : Nat) - (1 : Int)).toNat = bound := by omega
  rw [count, length, ← range_sum]
  have values : List.range (bound + 1) = 0 :: List.range' 1 bound := by
    rw [List.range_eq_range', List.range'_succ]
  rw [values]
  simp only [List.map_cons, endsAt89_zero, Bool.false_eq_true, if_false, List.sum_cons, Nat.zero_add]
  exact filter_sum _

/-- Selecting states that reach 89 yields exactly the reference's exclusive-bound count. -/
theorem Exact.answer {bound : Nat} {table : Table} (exact : Exact bound table) :
    weighted table (fun current => if endsAt89 current.1 then 1 else 0) = count (bound + 1 : Nat) := by
  rw [exact, count_succ]
  apply Finset.sum_congr rfl
  intro n member
  simp only [state, endsAt89_next]

end ProjectEulerSolutions.DigitSquareChains.Prefix
