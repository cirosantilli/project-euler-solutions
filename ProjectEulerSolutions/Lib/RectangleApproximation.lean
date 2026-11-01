import ProjectEulerSolutions.Lib.TriPentHex
import Mathlib.Data.Nat.Sqrt

/-! Exact triangular inversion and nearest rectangle-count search bounds. -/
namespace ProjectEulerSolutions.RectangleApproximation

set_option Elab.async false

open TriPentHex (triangular twice_triangular triangular_strictMono)

def count (m n : Nat) : Nat := triangular m * triangular n

def inverseTriangular (q : Nat) : Nat := (Nat.sqrt (1 + 8 * q) - 1) / 2

/-- The square-root expression is the greatest index below a triangular bound. -/
theorem le_inverseTriangular_iff (n q : Nat) :
    n ≤ inverseTriangular q ↔ triangular n ≤ q := by
  have positive : 0 < Nat.sqrt (1 + 8 * q) := Nat.sqrt_pos.mpr (by omega)
  have divide : n ≤ inverseTriangular q ↔ 2 * n + 1 ≤ Nat.sqrt (1 + 8 * q) := by
    unfold inverseTriangular
    omega
  rw [divide, Nat.le_sqrt]
  have twice := twice_triangular n
  constructor <;> intro h <;> nlinarith

theorem inverseTriangular_bounds (q : Nat) :
    triangular (inverseTriangular q) ≤ q ∧ q < triangular (inverseTriangular q + 1) := by
  constructor
  · exact (le_inverseTriangular_iff _ _).mp le_rfl
  · by_contra h
    have := (le_inverseTriangular_iff (inverseTriangular q + 1) q).mpr (by omega)
    omega

theorem count_formula (m n : Nat) :
    count m n = m * (m + 1) * n * (n + 1) / 4 := by
  have hm := twice_triangular m
  have hn := twice_triangular n
  have product : m * (m + 1) * n * (n + 1) = 4 * count m n := by
    unfold count
    nlinarith [congrArg (fun x => x * (2 * triangular n)) hm]
  rw [product]
  omega

theorem triangular_positive {m : Nat} (positive : 0 < m) : 0 < triangular m := by
  simpa [triangular] using triangular_strictMono positive

theorem count_strictMono {m : Nat} (positive : 0 < m) : StrictMono (count m) := by
  intro a b less
  exact Nat.mul_lt_mul_of_pos_left (triangular_strictMono less) (triangular_positive positive)

theorem count_row_lower {m n : Nat} (ordered : m ≤ n) : count m m ≤ count m n :=
  Nat.mul_le_mul_left _ (triangular_strictMono.monotone ordered)

def rowIndex (target : Int) (m : Nat) : Nat :=
  inverseTriangular (target.toNat / triangular m)

def rowBelow (target : Int) (m : Nat) : Nat := max m (rowIndex target m)

def distance (target : Int) (m n : Nat) : Int := |(count m n : Int) - target|

theorem le_rowIndex_iff (target : Int) {m : Nat} (positive : 0 < m) (n : Nat) :
    n ≤ rowIndex target m ↔ count m n ≤ target.toNat := by
  unfold rowIndex
  rw [le_inverseTriangular_iff, Nat.le_div_iff_mul_le (triangular_positive positive)]
  simp only [count, Nat.mul_comm]

theorem rowBelow_successor_above (target : Int) {m : Nat} (positive : 0 < m) :
    target < (count m (rowBelow target m + 1) : Int) := by
  have indexLess : rowIndex target m < rowBelow target m + 1 := by
    unfold rowBelow
    omega
  have above : target.toNat < count m (rowBelow target m + 1) := by
    by_contra h
    have := (le_rowIndex_iff target positive (rowBelow target m + 1)).mpr (by omega)
    omega
  omega

/-- Every earlier eligible column is strictly worse than the lower candidate. -/
theorem before_rowBelow (target : Int) {m n : Nat} (positive : 0 < m)
    (eligible : m ≤ n) (before : n < rowBelow target m) :
    distance target m (rowBelow target m) < distance target m n := by
  have below_eq : rowBelow target m = rowIndex target m := by
    unfold rowBelow at *
    omega
  have bounded : count m (rowBelow target m) ≤ target.toNat := by
    apply (le_rowIndex_iff target positive _).mp
    omega
  have strict := count_strictMono positive before
  have nonnegative : 0 ≤ target := by omega
  have upper : (count m (rowBelow target m) : Int) ≤ target := by omega
  have lower : (count m n : Int) < count m (rowBelow target m) := by exact_mod_cast strict
  unfold distance
  rw [abs_of_nonpos (by omega), abs_of_nonpos (by omega)]
  omega

/-- Every column after the upper candidate is strictly worse than that candidate. -/
theorem after_rowBelow (target : Int) {m n : Nat} (positive : 0 < m)
    (after : rowBelow target m + 1 < n) :
    distance target m (rowBelow target m + 1) < distance target m n := by
  have upper := rowBelow_successor_above target positive
  have strict := count_strictMono positive after
  have lower : (count m (rowBelow target m + 1) : Int) < count m n := by exact_mod_cast strict
  unfold distance
  rw [abs_of_nonneg (by omega), abs_of_nonneg (by omega)]
  omega

/-- Skipping all but these two columns cannot remove any distance minimizer. -/
theorem row_candidates_complete (target : Int) {m n : Nat} (positive : 0 < m)
    (eligible : m ≤ n) :
    n = rowBelow target m ∨ n = rowBelow target m + 1 ∨
      distance target m (rowBelow target m) < distance target m n ∨
      distance target m (rowBelow target m + 1) < distance target m n := by
  by_cases before : n < rowBelow target m
  · exact Or.inr (Or.inr (Or.inl (before_rowBelow target positive eligible before)))
  by_cases after : rowBelow target m + 1 < n
  · exact Or.inr (Or.inr (Or.inr (after_rowBelow target positive after)))
  omega

def bestColumn (target : Int) (m : Nat) : Nat :=
  let below := rowBelow target m
  if distance target m (below + 1) < distance target m below then below + 1 else below

theorem bestColumn_le_candidates (target : Int) (m : Nat) :
    distance target m (bestColumn target m) ≤ distance target m (rowBelow target m) ∧
    distance target m (bestColumn target m) ≤ distance target m (rowBelow target m + 1) := by
  dsimp only [bestColumn]
  split <;> constructor <;> omega

/-- The two-column comparison returns the least column among all row minimizers. -/
theorem bestColumn_spec (target : Int) {m : Nat} (positive : 0 < m) :
    m ≤ bestColumn target m ∧ ∀ n, m ≤ n →
      distance target m (bestColumn target m) ≤ distance target m n ∧
      (distance target m (bestColumn target m) = distance target m n → bestColumn target m ≤ n) := by
  have candidates := bestColumn_le_candidates target m
  have eligible : m ≤ rowBelow target m := Nat.le_max_left _ _
  constructor
  · dsimp only [bestColumn]
    split <;> omega
  intro n valid
  have complete := row_candidates_complete target positive valid
  constructor
  · rcases complete with h | h | h | h <;> (try subst n) <;> omega
  intro same
  have notBefore : rowBelow target m ≤ n := by
    by_contra h
    have := before_rowBelow target positive valid (by omega)
    omega
  by_cases atBelow : n = rowBelow target m
  · subst n
    dsimp only [bestColumn] at same ⊢
    split at same <;> split <;> omega
  · dsimp only [bestColumn]
    split <;> omega

/-- Python's nonnegative quotient clamp agrees with natural-number division. -/
theorem clamped_quotient (target : Int) {divisor : Nat} (positive : 0 < divisor) :
    max 0 (target / (divisor : Int)) = (target.toNat / divisor : Nat) := by
  by_cases nonnegative : 0 ≤ target
  · rw [Int.natCast_ediv, Int.toNat_of_nonneg nonnegative]
    exact max_eq_right (Int.ediv_nonneg nonnegative (by omega))
  · have negative := Int.ediv_neg_of_neg_of_pos (by omega : target < 0)
      (by exact_mod_cast positive : 0 < (divisor : Int))
    rw [max_eq_left (by omega), Int.toNat_eq_zero.mpr (by omega)]
    simp

theorem index_le_triangular (n : Nat) : n ≤ triangular n := by
  have twice := twice_triangular n
  nlinarith

theorem column_le_count {m : Nat} (positive : 0 < m) (n : Nat) : n ≤ count m n := by
  have factor := triangular_positive positive
  have lower := index_le_triangular n
  unfold count
  nlinarith

/-- The initial incumbent supplies a finite search bound for every signed target. -/
def searchBound (target : Int) : Nat := (target + |1 - target|).toNat

theorem active_column_bound (target difference : Int) {m n : Nat} (positive : 0 < m)
    (improved : difference ≤ |1 - target|)
    (active : (count m n : Int) ≤ target + difference) : n ≤ searchBound target := by
  have lower := column_le_count positive n
  unfold searchBound
  omega

theorem active_row_bound (target difference : Int) {m : Nat} (positive : 0 < m)
    (improved : difference ≤ |1 - target|)
    (active : (count m m : Int) ≤ target + difference) : m ≤ searchBound target :=
  active_column_bound target difference positive improved active

/-- Once the diagonal is too far above the target, every later row is worse. -/
theorem stop_excludes_later (target difference : Int) {m i n : Nat}
    (stop : target + difference < (count m m : Int)) (later : m ≤ i) (ordered : i ≤ n) :
    difference < distance target i n := by
  have hm := triangular_strictMono.monotone later
  have hn := triangular_strictMono.monotone (later.trans ordered)
  have lower : count m m ≤ count i n := Nat.mul_le_mul hm hn
  have absBound := le_abs_self ((count i n : Int) - target)
  unfold distance
  omega

end ProjectEulerSolutions.RectangleApproximation
