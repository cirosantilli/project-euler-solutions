import Mathlib.Data.Nat.Sqrt
import Mathlib.Tactic.Linarith
import Lean.Elab.Tactic.Omega
import ProjectEulerStatements.P42

/-! The square-root criterion for positive triangular numbers. -/
namespace ProjectEulerSolutions.TriangularDiscriminant

set_option Elab.async false

def triangular (k : Nat) : Nat := k * (k+1) / 2

def discriminant (n : Nat) : Bool :=
  if n = 0 then false else
    decide (Nat.sqrt (1+8*n) * Nat.sqrt (1+8*n) = 1+8*n ∧
      (Nat.sqrt (1+8*n)-1)%2 = 0)

private theorem twice_triangular (k : Nat) : 2 * triangular k = k * (k+1) := by
  have parity : k%2 = 0 ∨ k%2 = 1 := by omega
  have even : (k*(k+1))%2=0 := by
    rcases parity with h | h <;> simp [Nat.mul_mod, Nat.add_mod, h]
  have divided := Nat.mod_add_div (k*(k+1)) 2
  simp only [even, Nat.zero_add] at divided
  exact divided

theorem square_iff (n : Nat) (positive : 0 < n) :
    (Nat.sqrt (1+8*n) * Nat.sqrt (1+8*n) = 1+8*n ∧
      (Nat.sqrt (1+8*n)-1)%2 = 0) ↔
    ∃ k, k ≤ n ∧ triangular k = n := by
  constructor
  · rintro ⟨square, even⟩
    let root := Nat.sqrt (1+8*n)
    let k := (root-1)/2
    have rootPositive : 0 < root := by dsimp [root]; nlinarith
    have decomposed : root = 2*k+1 := by dsimp [k]; omega
    have product : k*(k+1)=2*n := by
      change root*root=1+8*n at square
      rw [decomposed] at square
      nlinarith
    have kPositive : 0 < k := by nlinarith
    have bound : k ≤ n := by nlinarith [Nat.le_mul_self k]
    refine ⟨k, bound, ?_⟩
    simp [triangular, product]
  · rintro ⟨k, _, found⟩
    have product := twice_triangular k
    rw [found] at product
    have square : 1+8*n=(2*k+1)*(2*k+1) := by nlinarith
    rw [square, Nat.sqrt_eq]
    constructor
    · rfl
    · omega

theorem exhaustive_iff (n : Nat) :
    ProjectEulerStatements.P42.isTriangle n = true ↔ ∃ k, k ≤ n ∧ triangular k = n := by
  simp [ProjectEulerStatements.P42.isTriangle, List.any_eq_true, List.mem_range,
    Nat.lt_succ_iff, triangular]

theorem correct (n : Nat) (positive : 0 < n) :
    discriminant n = ProjectEulerStatements.P42.isTriangle n := by
  apply Bool.eq_iff_iff.mpr
  rw [exhaustive_iff]
  simpa only [discriminant, if_neg (by omega : n ≠ 0), decide_eq_true_eq] using square_iff n positive

end ProjectEulerSolutions.TriangularDiscriminant
