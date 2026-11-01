import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Nat.Sqrt
import Mathlib.Data.Finset.Union
import Mathlib.Data.Finset.Image
import Mathlib.Data.List.TakeWhile
import Mathlib.Tactic

/-! Complete prime bounds and distinct sums for square/cube/fourth-power searches. -/
namespace ProjectEulerSolutions.PrimePowerTriples
set_option Elab.async false

def primesThrough (bound : Nat) : Finset Nat := (Finset.range (bound + 1)).filter Nat.Prime

@[simp] theorem mem_primesThrough (bound p : Nat) :
    p ∈ primesThrough bound ↔ Nat.Prime p ∧ p ≤ bound := by
  simp [primesThrough, and_comm]

def Represents (value : Nat) : Prop :=
  ∃ p q r : Nat, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ value = p ^ 2 + q ^ 3 + r ^ 4

theorem square_le_power (p exponent : Nat) (positive : 1 ≤ p) (degree : 2 ≤ exponent) :
    p ^ 2 ≤ p ^ exponent := Nat.pow_le_pow_right positive degree

theorem candidate_bound (limit p exponent : Nat) (positive : 1 ≤ p) (degree : 2 ≤ exponent)
    (inside : p ^ exponent < limit) : p ≤ Nat.sqrt limit := by
  apply Nat.le_sqrt'.mpr
  exact le_trans (square_le_power p exponent positive degree) (Nat.le_of_lt inside)

def exhaustive (limit : Nat) : Finset Nat :=
  (primesThrough (Nat.sqrt limit)).biUnion fun p =>
    (primesThrough (Nat.sqrt limit)).biUnion fun q =>
      ((primesThrough (Nat.sqrt limit)).image (fun r => p ^ 2 + q ^ 3 + r ^ 4)).filter (· < limit)

theorem mem_exhaustive (limit value : Nat) :
    value ∈ exhaustive limit ↔ value < limit ∧ Represents value := by
  constructor
  · intro member
    obtain ⟨p, hp, q, hq, ⟨r, hr, equal⟩, inside⟩ := by
      simpa only [exhaustive, Finset.mem_biUnion, Finset.mem_filter, Finset.mem_image] using member
    exact ⟨inside, p, q, r, (mem_primesThrough _ _).mp hp |>.1,
      (mem_primesThrough _ _).mp hq |>.1, (mem_primesThrough _ _).mp hr |>.1, equal.symm⟩
  · rintro ⟨inside, p, q, r, hp, hq, hr, rfl⟩
    have bp := candidate_bound limit p 2 (by have := hp.two_le; omega) (by omega) (by omega)
    have bq := candidate_bound limit q 3 (by have := hq.two_le; omega) (by omega) (by omega)
    have br := candidate_bound limit r 4 (by have := hr.two_le; omega) (by omega) (by omega)
    simp only [exhaustive, Finset.mem_biUnion, Finset.mem_filter, Finset.mem_image]
    exact ⟨p, (mem_primesThrough _ _).mpr ⟨hp, bp⟩,
      q, (mem_primesThrough _ _).mpr ⟨hq, bq⟩,
      ⟨r, (mem_primesThrough _ _).mpr ⟨hr, br⟩, rfl⟩, inside⟩

def powers (limit exponent : Nat) : Finset Nat :=
  ((primesThrough (Nat.sqrt limit + 1)).image (· ^ exponent)).filter (· < limit)

theorem mem_powers (limit exponent value : Nat) (degree : 2 ≤ exponent) :
    value ∈ powers limit exponent ↔ value < limit ∧ ∃ p, Nat.Prime p ∧ value = p ^ exponent := by
  constructor
  · intro member
    obtain ⟨⟨p, hp, equal⟩, inside⟩ := by
      simpa only [powers, Finset.mem_filter, Finset.mem_image] using member
    exact ⟨inside, p, ((mem_primesThrough _ _).mp hp).1, equal.symm⟩
  · rintro ⟨inside, p, hp, rfl⟩
    have bound := candidate_bound limit p exponent (by have := hp.two_le; omega) degree inside
    simp only [powers, Finset.mem_filter, Finset.mem_image]
    exact ⟨⟨p, (mem_primesThrough _ _).mpr ⟨hp, by omega⟩, rfl⟩, inside⟩

def grouped (limit : Nat) : Finset Nat :=
  (powers limit 4).biUnion fun f => (powers limit 3).biUnion fun c =>
    ((powers limit 2).image (fun s => f + c + s)).filter (· < limit)

theorem mem_grouped (limit value : Nat) :
    value ∈ grouped limit ↔ value < limit ∧ Represents value := by
  constructor
  · intro member
    obtain ⟨f, hf, c, hc, ⟨s, hs, equal⟩, inside⟩ := by
      simpa only [grouped, Finset.mem_biUnion, Finset.mem_filter, Finset.mem_image] using member
    obtain ⟨_, r, hr, rfl⟩ := (mem_powers limit 4 f (by omega)).mp hf
    obtain ⟨_, q, hq, rfl⟩ := (mem_powers limit 3 c (by omega)).mp hc
    obtain ⟨_, p, hp, rfl⟩ := (mem_powers limit 2 s (by omega)).mp hs
    exact ⟨inside, p, q, r, hp, hq, hr, by omega⟩
  · rintro ⟨inside, p, q, r, hp, hq, hr, rfl⟩
    have hs := (mem_powers limit 2 (p ^ 2) (by omega)).mpr ⟨by omega, p, hp, rfl⟩
    have hc := (mem_powers limit 3 (q ^ 3) (by omega)).mpr ⟨by omega, q, hq, rfl⟩
    have hf := (mem_powers limit 4 (r ^ 4) (by omega)).mpr ⟨by omega, r, hr, rfl⟩
    simp only [grouped, Finset.mem_biUnion, Finset.mem_filter, Finset.mem_image]
    exact ⟨r ^ 4, hf, q ^ 3, hc, ⟨p ^ 2, hs, by omega⟩, inside⟩

theorem grouping_equivalence (limit : Nat) : exhaustive limit = grouped limit := by
  ext value
  rw [mem_exhaustive, mem_grouped]

theorem empty_nonpositive (limit : Int) (nonpositive : limit ≤ 0) : exhaustive limit.toNat = ∅ := by
  ext value
  simp [mem_exhaustive, Int.toNat_eq_zero.mpr nonpositive]

/-- Sorted powers allow a break at the first failing value without losing later solutions. -/
theorem sorted_prefix (limit offset : Nat) (items : List Nat) (ordered : items.Pairwise (· ≤ ·)) :
    items.takeWhile (fun x => decide (offset + x < limit)) =
      items.filter (fun x => decide (offset + x < limit)) := by
  induction items with
  | nil => rfl
  | cons head tail ih =>
      have pair := List.pairwise_cons.mp ordered
      by_cases active : offset + head < limit
      · simp [active, ih pair.2]
      · have empty : tail.filter (fun x => decide (offset + x < limit)) = [] := by
          apply List.filter_eq_nil_iff.mpr
          intro x member
          have later := pair.1 x member
          simp only [Bool.not_eq_true, decide_eq_false_iff_not]
          omega
        simp [active, empty]

end ProjectEulerSolutions.PrimePowerTriples
