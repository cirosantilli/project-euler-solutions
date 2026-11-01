import ProjectEulerSolutions.Lib.LatticePerpendicular
import Mathlib.Data.Int.Interval
import Mathlib.Data.Finset.Prod

/-! Exact signed step intervals for perpendicular lines inside a square. -/
namespace ProjectEulerSolutions.LatticeRightTriangles
set_option Elab.async false

def grid (n : Nat) : Finset (Int × Int) := Finset.Icc 0 (n : Int) ×ˢ Finset.Icc 0 (n : Int)

theorem mem_grid (n : Nat) (p : Int × Int) :
    p ∈ grid n ↔ 0 ≤ p.1 ∧ p.1 ≤ n ∧ 0 ≤ p.2 ∧ p.2 ≤ n := by
  simp only [grid, Finset.mem_product, Finset.mem_Icc]
  tauto

theorem signed_step_bounds (n x y dx dy k : Int) (positiveX : 0 < dx) (positiveY : 0 < dy) :
    (0 ≤ x - dx * k ∧ x - dx * k ≤ n ∧ 0 ≤ y + dy * k ∧ y + dy * k ≤ n) ↔
      -min ((n - x) / dx) (y / dy) ≤ k ∧ k ≤ min (x / dx) ((n - y) / dy) := by
  rw [neg_le, le_min_iff, le_min_iff]
  simp only [Int.le_ediv_iff_mul_le positiveX, Int.le_ediv_iff_mul_le positiveY, neg_mul]
  simp only [mul_comm k]
  omega

def upperSteps (n x y : Nat) : Nat := min (x / (y / x.gcd y)) ((n - y) / (x / x.gcd y))
def lowerSteps (n x y : Nat) : Nat := min ((n - x) / (y / x.gcd y)) (y / (x / x.gcd y))

def stepPoint (x y : Nat) (k : Int) : Int × Int :=
  ((x : Int) - (y / x.gcd y : Nat) * k, (y : Int) + (x / x.gcd y : Nat) * k)

theorem stepPoint_in_grid (n x y : Nat) (hx : 0 < x) (hy : 0 < y)
    (xBound : x ≤ n) (yBound : y ≤ n) (k : Int) :
    stepPoint x y k ∈ grid n ↔ -(lowerSteps n x y : Int) ≤ k ∧ k ≤ (upperSteps n x y : Int) := by
  have dx : (0 : Int) < (y / x.gcd y : Nat) := by
    exact_mod_cast (by simpa [Nat.gcd_comm] using divided_gcd_positive y x hy)
  have dy : (0 : Int) < (x / x.gcd y : Nat) := by
    exact_mod_cast divided_gcd_positive x y hx
  rw [mem_grid]
  simp only [stepPoint]
  rw [signed_step_bounds _ _ _ _ _ _ dx dy]
  simp only [lowerSteps, upperSteps, Nat.cast_min, Int.natCast_ediv,
    Int.natCast_sub xBound, Int.natCast_sub yBound]

theorem stepPoint_injective (x y : Nat) (hx : 0 < x) : Function.Injective (stepPoint x y) := by
  intro k l same
  have second := congrArg Prod.snd same
  have nonzero : ((x / x.gcd y : Nat) : Int) ≠ 0 := by
    exact_mod_cast Nat.ne_of_gt (divided_gcd_positive x y hx)
  apply mul_left_cancel₀ nonzero
  exact add_left_cancel second

theorem stepPoint_eq_self (x y : Nat) (hx : 0 < x) (k : Int) :
    stepPoint x y k = ((x : Int), (y : Int)) ↔ k = 0 := by
  change stepPoint x y k = stepPoint x y 0 ↔ _
  exact (stepPoint_injective x y hx).eq_iff

def admissibleSteps (lower upper : Nat) : Finset Int :=
  (Finset.Icc (-(lower : Int)) (upper : Int)).erase 0

theorem mem_admissibleSteps (lower upper : Nat) (k : Int) :
    k ∈ admissibleSteps lower upper ↔ k ≠ 0 ∧ -(lower : Int) ≤ k ∧ k ≤ (upper : Int) := by
  simp [admissibleSteps]

theorem admissibleSteps_card (lower upper : Nat) :
    (admissibleSteps lower upper).card = lower + upper := by
  rw [admissibleSteps, Finset.card_erase_of_mem (by simp), Int.card_Icc]
  omega

end ProjectEulerSolutions.LatticeRightTriangles
