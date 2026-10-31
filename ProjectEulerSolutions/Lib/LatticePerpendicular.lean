import ProjectEulerSolutions.Lib.LatticeRightTriangles
import Mathlib.RingTheory.Coprime.Lemmas

/-! Primitive perpendicular steps enumerate every integral displacement, without a search bound. -/
namespace ProjectEulerSolutions.LatticeRightTriangles

theorem primitive_perpendicular (u v r s : Int) (nonzero : u ≠ 0) (coprime : IsCoprime u v) :
    u * r + v * s = 0 ↔ ∃ k : Int, r = -v * k ∧ s = u * k := by
  constructor
  · intro orthogonal
    have divides : u ∣ v * s := ⟨-r, by nlinarith⟩
    obtain ⟨k, hk⟩ := coprime.dvd_of_dvd_mul_left divides
    refine ⟨k, ?_, hk⟩
    apply mul_left_cancel₀ nonzero
    rw [hk] at orthogonal
    nlinarith only [orthogonal]
  · rintro ⟨k, rfl, rfl⟩
    ring

theorem perpendicular_step_unique (u s k l : Int) (nonzero : u ≠ 0)
    (first : s = u * k) (second : s = u * l) : k = l :=
  mul_left_cancel₀ nonzero (first.symm.trans second)

theorem scaled_orthogonal (g u v r s : Int) (nonzero : g ≠ 0) :
    (g * u) * r + (g * v) * s = 0 ↔ u * r + v * s = 0 := by
  rw [show (g * u) * r + (g * v) * s = g * (u * r + v * s) by ring, mul_eq_zero]
  simp [nonzero]

theorem divided_gcd_positive (x y : Nat) (positive : 0 < x) : 0 < x / x.gcd y :=
  Nat.div_pos (Nat.gcd_le_left y positive) (Nat.gcd_pos_of_pos_left y positive)

/-- Every integer perpendicular displacement is an integer multiple of the reduced direction. -/
theorem gcd_perpendicular (x y : Nat) (positive : 0 < x) (r s : Int) :
    (x : Int) * r + (y : Int) * s = 0 ↔
      ∃ k : Int, r = -(y / x.gcd y : Nat) * k ∧ s = (x / x.gcd y : Nat) * k := by
  have gPositive := Nat.gcd_pos_of_pos_left y positive
  have xFactor : (x : Int) = (x.gcd y : Int) * (x / x.gcd y : Nat) := by
    exact_mod_cast (Nat.mul_div_cancel' (Nat.gcd_dvd_left x y)).symm
  have yFactor : (y : Int) = (x.gcd y : Int) * (y / x.gcd y : Nat) := by
    exact_mod_cast (Nat.mul_div_cancel' (Nat.gcd_dvd_right x y)).symm
  have nonzero : (x.gcd y : Int) ≠ 0 := by exact_mod_cast Nat.ne_of_gt gPositive
  have uNonzero : (x / x.gcd y : Nat) ≠ 0 := Nat.ne_of_gt (divided_gcd_positive x y positive)
  have coprime : Nat.Coprime (x / x.gcd y) (y / x.gcd y) :=
    Nat.gcd_div_gcd_div_gcd_of_pos_left positive
  conv_lhs => rw [xFactor, yFactor, scaled_orthogonal _ _ _ _ _ nonzero]
  exact primitive_perpendicular _ _ r s (by exact_mod_cast uNonzero) coprime.isCoprime

/-- The integer step is zero exactly when the second point is the first point. -/
theorem gcd_step_zero (x y : Nat) (positive : 0 < x) (k : Int) :
    (x / x.gcd y : Nat) * k = (0 : Int) ↔ k = 0 := by
  have nonzero : ((x / x.gcd y : Nat) : Int) ≠ 0 := by
    have h := divided_gcd_positive x y positive
    exact_mod_cast (Nat.ne_of_gt h)
  rw [mul_eq_zero]
  simp only [nonzero, false_or]

/-- The primitive direction parametrizes the whole perpendicular line through a grid point. -/
theorem perpendicular_points (x y : Nat) (positive : 0 < x) (c d : Int) :
    atFirst x y c d ↔ ∃ k : Int,
      c = (x : Int) - (y / x.gcd y : Nat) * k ∧ d = (y : Int) + (x / x.gcd y : Nat) * k := by
  rw [atFirst, gcd_perpendicular x y positive]
  constructor
  · rintro ⟨k, hc, hd⟩
    refine ⟨k, ?_, ?_⟩ <;> dsimp at * <;> nlinarith
  · rintro ⟨k, hc, hd⟩
    refine ⟨k, ?_, ?_⟩ <;> dsimp at * <;> nlinarith

/-- The minimum of the two quotient bounds counts exactly the steps fitting in one direction. -/
theorem step_bound (n x y dx dy k : Nat) (inside : y ≤ n) (positiveX : 0 < dx) (positiveY : 0 < dy) :
    dx * k ≤ x ∧ y + dy * k ≤ n ↔ k ≤ min (x / dx) ((n - y) / dy) := by
  simp only [le_min_iff, Nat.le_div_iff_mul_le positiveX, Nat.le_div_iff_mul_le positiveY]
  rw [Nat.le_sub_iff_add_le inside]
  simp only [Nat.mul_comm, Nat.add_comm]

end ProjectEulerSolutions.LatticeRightTriangles
