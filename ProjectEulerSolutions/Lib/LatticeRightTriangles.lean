import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Squared-distance tests and the unique right-angle vertex of a lattice triangle. -/
namespace ProjectEulerSolutions.LatticeRightTriangles

def axisDistance (a b : Int) : Int := if a ≤ b then b - a else a - b

def norm (a b : Int) : Int := a * a + b * b

def distance (a b c d : Int) : Int :=
  let dx := axisDistance a c
  let dy := axisDistance b d
  dx * dx + dy * dy

def isRight (a b c d : Int) : Bool :=
  decide (norm a b + distance a b c d = norm c d ∨
    norm a b + norm c d = distance a b c d ∨
    distance a b c d + norm c d = norm a b)

def atOrigin (a b c d : Int) : Prop := a * c + b * d = 0
def atFirst (a b c d : Int) : Prop := a * (c - a) + b * (d - b) = 0
def atSecond (a b c d : Int) : Prop := c * (a - c) + d * (b - d) = 0

instance (a b c d : Int) : Decidable (atOrigin a b c d) := inferInstanceAs (Decidable (_ = _))
instance (a b c d : Int) : Decidable (atFirst a b c d) := inferInstanceAs (Decidable (_ = _))
instance (a b c d : Int) : Decidable (atSecond a b c d) := inferInstanceAs (Decidable (_ = _))

theorem axisDistance_square (a b : Int) : axisDistance a b * axisDistance a b = (b - a) ^ 2 := by
  unfold axisDistance
  split <;> ring

theorem distance_eq (a b c d : Int) : distance a b c d = (c - a) ^ 2 + (d - b) ^ 2 := by
  simp only [distance, axisDistance_square]

theorem norm_nonnegative (a b : Int) : 0 ≤ norm a b := by
  unfold norm
  nlinarith [sq_nonneg a, sq_nonneg b]

theorem norm_zero (a b : Int) : norm a b = 0 ↔ a = 0 ∧ b = 0 := by
  constructor
  · intro h
    unfold norm at h
    constructor <;> nlinarith [sq_nonneg a, sq_nonneg b]
  · rintro ⟨rfl, rfl⟩
    rfl

theorem first_iff_lengths (a b c d : Int) :
    atFirst a b c d ↔ norm a b + distance a b c d = norm c d := by
  simp only [atFirst, norm, distance_eq]
  constructor <;> intro h <;> nlinarith

theorem origin_iff_lengths (a b c d : Int) :
    atOrigin a b c d ↔ norm a b + norm c d = distance a b c d := by
  simp only [atOrigin, norm, distance_eq]
  constructor <;> intro h <;> nlinarith

theorem second_iff_lengths (a b c d : Int) :
    atSecond a b c d ↔ distance a b c d + norm c d = norm a b := by
  simp only [atSecond, norm, distance_eq]
  constructor <;> intro h <;> nlinarith

/-- The three Pythagorean equalities test the three possible right-angle vertices. -/
theorem isRight_iff (a b c d : Int) :
    isRight a b c d = true ↔ atFirst a b c d ∨ atOrigin a b c d ∨ atSecond a b c d := by
  simp only [isRight, decide_eq_true_eq, ← first_iff_lengths, ← origin_iff_lengths, ← second_iff_lengths]

theorem isRight_swap (a b c d : Int) : isRight a b c d = isRight c d a b := by
  apply Bool.eq_iff_iff.mpr
  simp only [isRight_iff, atFirst, atOrigin, atSecond]
  constructor <;> rintro (first | origin | second)
  · exact Or.inr (Or.inr first)
  · exact Or.inr (Or.inl (by nlinarith))
  · exact Or.inl second
  · exact Or.inr (Or.inr first)
  · exact Or.inr (Or.inl (by nlinarith))
  · exact Or.inl second

theorem origin_first_exclusive (a b c d : Int) (nonzero : a ≠ 0 ∨ b ≠ 0)
    (origin : atOrigin a b c d) : ¬ atFirst a b c d := by
  intro first
  have zero : norm a b = 0 := by
    unfold norm
    unfold atOrigin at origin
    unfold atFirst at first
    nlinarith
  obtain ⟨ha, hb⟩ := (norm_zero a b).mp zero
  rcases nonzero with h | h <;> contradiction

theorem origin_second_exclusive (a b c d : Int) (nonzero : c ≠ 0 ∨ d ≠ 0)
    (origin : atOrigin a b c d) : ¬ atSecond a b c d := by
  exact origin_first_exclusive c d a b nonzero (by simpa [atOrigin, mul_comm] using origin)

theorem first_second_exclusive (a b c d : Int) (distinct : a ≠ c ∨ b ≠ d)
    (first : atFirst a b c d) : ¬ atSecond a b c d := by
  intro second
  have zero : norm (c - a) (d - b) = 0 := by
    unfold norm
    unfold atFirst at first
    unfold atSecond at second
    nlinarith
  obtain ⟨ha, hb⟩ := (norm_zero (c - a) (d - b)).mp zero
  rcases distinct with h | h <;> apply h <;> omega

/-- On nondegenerate triangles the disjunction counts exactly one right-angle vertex. -/
theorem right_indicator (a b c d : Int)
    (firstNonzero : a ≠ 0 ∨ b ≠ 0) (secondNonzero : c ≠ 0 ∨ d ≠ 0)
    (distinct : a ≠ c ∨ b ≠ d) :
    (if isRight a b c d then 1 else 0 : Nat) =
      (if atOrigin a b c d then 1 else 0) + (if atFirst a b c d then 1 else 0) +
        (if atSecond a b c d then 1 else 0) := by
  have h1 := origin_first_exclusive a b c d firstNonzero
  have h2 := origin_second_exclusive a b c d secondNonzero
  have h3 := first_second_exclusive a b c d distinct
  by_cases ho : atOrigin a b c d <;> by_cases hp : atFirst a b c d <;>
    by_cases hq : atSecond a b c d <;> simp_all [isRight_iff]

end ProjectEulerSolutions.LatticeRightTriangles
