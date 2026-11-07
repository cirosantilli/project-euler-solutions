import Mathlib.NumberTheory.Pell
import Mathlib.Tactic

/-! Exact integer geometry and the complete Pell family of almost-equilateral triangles. -/
namespace ProjectEulerSolutions.AlmostEquilateral

set_option Elab.async false

def fundamental : Pell.Solution₁ 3 := Pell.Solution₁.mk 2 1 (by norm_num)

theorem fundamental_isFundamental : Pell.IsFundamental fundamental := by
  refine ⟨by norm_num [fundamental], by norm_num [fundamental], ?_⟩
  intro other larger
  change 2 ≤ other.x
  omega

def solution (n : Nat) : Pell.Solution₁ 3 := fundamental ^ n

theorem solution_zero : solution 0 = 1 := rfl

theorem solution_next_x (n : Nat) :
    (solution (n + 1)).x = 2 * (solution n).x + 3 * (solution n).y := by
  simp [solution, pow_succ, Pell.Solution₁.x_mul, fundamental]
  ring

theorem solution_next_y (n : Nat) :
    (solution (n + 1)).y = (solution n).x + 2 * (solution n).y := by
  simp [solution, pow_succ, Pell.Solution₁.y_mul, fundamental]
  ring

theorem solution_positive (n : Nat) : 0 < (solution n).x ∧ 0 ≤ (solution n).y := by
  induction n with
  | zero => norm_num [solution]
  | succ n ih => rw [solution_next_x, solution_next_y]; omega

theorem solution_x_strictMono : StrictMono fun n => (solution n).x := by
  apply strictMono_nat_of_lt_succ
  intro n
  rw [solution_next_x]
  have := solution_positive n
  omega

theorem solution_complete (x y : Int) (positive : 0 < x) (nonnegative : 0 ≤ y)
    (equation : x ^ 2 - 3 * y ^ 2 = 1) :
    ∃ n, (solution n).x = x ∧ (solution n).y = y := by
  obtain ⟨n, same⟩ := fundamental_isFundamental.eq_pow_of_nonneg
    (a := Pell.Solution₁.mk x y equation) positive nonnegative
  exact ⟨n, congrArg Pell.Solution₁.x same.symm, congrArg Pell.Solution₁.y same.symm⟩

/-- Removing a nonzero square factor preserves integral squareness. -/
theorem square_factor (h value : Int) (nonzero : h ≠ 0) :
    IsSquare (h ^ 2 * value) ↔ IsSquare value := by
  constructor
  · rintro ⟨area, squared⟩
    have divides : h ^ 2 ∣ area ^ 2 := ⟨value, by nlinarith [squared]⟩
    obtain ⟨height, height_eq⟩ := (Int.pow_dvd_pow_iff (by decide : (2 : Nat) ≠ 0)).mp divides
    refine ⟨height, ?_⟩
    have same : h ^ 2 * value = h ^ 2 * (height * height) := by rw [squared, height_eq]; ring
    exact mul_left_cancel₀ (pow_ne_zero 2 nonzero) same
  · exact fun square => ((isSquare_iff_exists_sq _).mpr ⟨h, rfl⟩).mul square

/-- Heron's integer test for positive isosceles triangles with an even base. -/
theorem heron_iff_height (a h : Int) (positive : 0 < h) (triangle : h < a) :
    IsSquare ((a + h) * h * h * (a - h)) ↔
      ∃ height : Int, 0 < height ∧ a ^ 2 = h ^ 2 + height ^ 2 := by
  rw [show (a + h) * h * h * (a - h) = h ^ 2 * (a ^ 2 - h ^ 2) by ring,
    square_factor h _ (by omega)]
  constructor
  · rintro ⟨height, square⟩
    refine ⟨|height|, ?_, ?_⟩
    · rw [abs_pos]
      intro zero
      subst height
      nlinarith
    · rw [sq_abs]
      nlinarith
  · rintro ⟨height, _, equation⟩
    exact ⟨height, by nlinarith⟩

/-- A triangle with base one longer than its equal sides determines a positive Pell solution. -/
theorem longer_to_pell (a h height : Int) (base : 2 * h = a + 1)
    (area : a ^ 2 = h ^ 2 + height ^ 2) :
    (3 * h - 2) ^ 2 - 3 * height ^ 2 = 1 := by nlinarith

/-- A triangle with base one shorter than its equal sides determines a positive Pell solution. -/
theorem shorter_to_pell (a h height : Int) (base : 2 * h = a - 1)
    (area : a ^ 2 = h ^ 2 + height ^ 2) :
    (3 * h + 2) ^ 2 - 3 * height ^ 2 = 1 := by nlinarith

/-- Exact positive, nondegenerate almost-equilateral triangles with integral height. -/
def Triangle (a b : Int) : Prop :=
  1 < a ∧ (b = a + 1 ∨ b = a - 1) ∧
    ∃ h height : Int, 0 < h ∧ 0 < height ∧ b = 2 * h ∧ a ^ 2 = h ^ 2 + height ^ 2

def triangle (n : Nat) : Int × Int :=
  let x := (solution n).x
  if x % 3 = 1 then (2 * ((x + 2) / 3) - 1, 2 * ((x + 2) / 3))
  else (2 * ((x - 2) / 3) + 1, 2 * ((x - 2) / 3))

theorem solution_mod_three (n : Nat) : (solution n).x % 3 = 1 ∨ (solution n).x % 3 = 2 := by
  induction n with
  | zero => norm_num [solution]
  | succ n ih => rw [solution_next_x]; omega

theorem solution_large (n : Nat) (nontrivial : 2 ≤ n) :
    7 ≤ (solution n).x ∧ 0 < (solution n).y := by
  have lower := solution_x_strictMono.monotone nontrivial
  have initial : (solution 2).x = 7 := by norm_num [solution, pow_succ, Pell.Solution₁.x_mul, fundamental]
  rw [initial] at lower
  have positive := solution_positive n
  have equation := (solution n).prop
  exact ⟨lower, by nlinarith⟩

/-- Every term after the two degenerate initial solutions gives a valid triangle. -/
theorem triangle_valid (n : Nat) (nontrivial : 2 ≤ n) :
    Triangle (triangle n).1 (triangle n).2 := by
  obtain ⟨large, positive⟩ := solution_large n nontrivial
  have equation := (solution n).prop
  have residues := solution_mod_three n
  unfold triangle
  dsimp only
  split_ifs with residue
  · have divided : 3 * (((solution n).x + 2) / 3) = (solution n).x + 2 := by omega
    refine ⟨by omega, Or.inl (by omega), ((solution n).x + 2) / 3, (solution n).y,
      by omega, positive, rfl, ?_⟩
    nlinarith
  · have divided : 3 * (((solution n).x - 2) / 3) = (solution n).x - 2 := by omega
    refine ⟨by omega, Or.inr (by omega), ((solution n).x - 2) / 3, (solution n).y,
      by omega, positive, rfl, ?_⟩
    nlinarith

theorem solution_nontrivial (n : Nat) (large : 2 < (solution n).x) : 2 ≤ n := by
  by_contra small
  have : n = 0 ∨ n = 1 := by omega
  rcases this with rfl | rfl <;> norm_num [solution, fundamental] at large

/-- Completeness: no positive integral triangle is skipped by the Pell family. -/
theorem triangle_complete (a b : Int) (valid : Triangle a b) :
    ∃ n, 2 ≤ n ∧ triangle n = (a, b) := by
  obtain ⟨positiveA, base, h, height, positiveH, positiveHeight, equalBase, area⟩ := valid
  rcases base with longer | shorter
  · have base : 2 * h = a + 1 := by omega
    obtain ⟨n, x, y⟩ := solution_complete (3 * h - 2) height (by omega) (by omega)
      (longer_to_pell a h height base area)
    refine ⟨n, solution_nontrivial n (by rw [x]; omega), ?_⟩
    have residue : (3 * h - 2) % 3 = 1 := by omega
    simp only [triangle, x, residue, ↓reduceIte]
    have divided : (3 * h - 2 + 2) / 3 = h := by omega
    rw [divided]
    congr 1 <;> omega
  · have base : 2 * h = a - 1 := by omega
    obtain ⟨n, x, y⟩ := solution_complete (3 * h + 2) height (by omega) (by omega)
      (shorter_to_pell a h height base area)
    refine ⟨n, solution_nontrivial n (by rw [x]; omega), ?_⟩
    have residue : (3 * h + 2) % 3 ≠ 1 := by omega
    simp only [triangle, x, residue, ↓reduceIte]
    have divided : (3 * h + 2 - 2) / 3 = h := by omega
    rw [divided]
    congr 1 <;> omega

def heron (a b : Int) : Int :=
  let p := (2 * a + b) / 2
  p * (p - a) * (p - a) * (p - b)

/-- The reference's semiperimeter and integer-square test express the same exact geometry. -/
theorem triangle_iff_heron (a b : Int) :
    Triangle a b ↔ 1 < a ∧ (b = a + 1 ∨ b = a - 1) ∧
      (2 * a + b) % 2 = 0 ∧ IsSquare (heron a b) := by
  constructor
  · rintro ⟨positiveA, base, h, height, positiveH, positiveHeight, equalBase, area⟩
    have less : h < a := by nlinarith
    have half : (2 * a + b) / 2 = a + h := by omega
    refine ⟨positiveA, base, by omega, ?_⟩
    have square := (heron_iff_height a h positiveH less).mpr ⟨height, positiveHeight, area⟩
    dsimp only [heron]
    rw [half]
    simpa only [equalBase, show a + h - a = h by omega,
      show a + h - 2 * h = a - h by omega] using square
  · rintro ⟨positiveA, base, even, square⟩
    let h := b / 2
    have equalBase : b = 2 * h := by dsimp [h]; omega
    have positiveH : 0 < h := by rcases base with base | base <;> omega
    have less : h < a := by rcases base with base | base <;> omega
    have half : (2 * a + b) / 2 = a + h := by omega
    have square' : IsSquare ((a + h) * h * h * (a - h)) := by
      dsimp only [heron] at square
      rw [half] at square
      simpa only [equalBase, show a + h - a = h by omega,
        show a + h - 2 * h = a - h by omega] using square
    obtain ⟨height, positiveHeight, area⟩ := (heron_iff_height a h positiveH less).mp square'
    exact ⟨positiveA, base, h, height, positiveH, positiveHeight, equalBase, area⟩

instance (a b : Int) : Decidable (Triangle a b) :=
  decidable_of_iff _ (triangle_iff_heron a b).symm

def perimeter (n : Nat) : Int := 2 * (triangle n).1 + (triangle n).2

theorem perimeter_eq (n : Nat) : perimeter n =
    if (solution n).x % 3 = 1 then 2 * (solution n).x + 2 else 2 * (solution n).x - 2 := by
  have residues := solution_mod_three n
  unfold perimeter triangle
  dsimp only
  split_ifs <;> dsimp only <;> omega

theorem perimeter_bounds (n : Nat) :
    2 * (solution n).x - 2 ≤ perimeter n ∧ perimeter n ≤ 2 * (solution n).x + 2 := by
  rw [perimeter_eq]
  split_ifs <;> omega

theorem perimeter_increasing (n : Nat) (nontrivial : 2 ≤ n) : perimeter n < perimeter (n + 1) := by
  have first := perimeter_bounds n
  have next := perimeter_bounds (n + 1)
  have large := solution_large n nontrivial
  have recurrence := solution_next_x n
  omega

/-- The perimeter calculated by the actual Python branch, before its limit test. -/
def sourcePerimeter (x : Int) : Int :=
  if x % 3 = 2 then 3 * ((x + 1) / 3) + 1 else 3 * ((x - 1) / 3) - 1

theorem sourcePerimeter_eq (n : Nat) : sourcePerimeter (2 * (solution n).x) = perimeter n := by
  have residues := solution_mod_three n
  rw [perimeter_eq]
  unfold sourcePerimeter
  split_ifs <;> omega

/-- The source loop stops at the first excessive perimeter, with no iteration budget. -/
def perimetersFrom (limit : Int) (n : Nat) (nontrivial : 2 ≤ n) : List Int :=
  if limit < perimeter n then []
  else perimeter n :: perimetersFrom limit (n + 1) (by omega)
termination_by (limit + 3 - 2 * (solution n).x).toNat
decreasing_by
  have bounds := perimeter_bounds n
  have growth : (solution n).x < (solution (n + 1)).x := solution_x_strictMono (Nat.lt_succ_self n)
  omega

end ProjectEulerSolutions.AlmostEquilateral
