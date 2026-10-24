import Mathlib.Data.Nat.Sqrt
import Mathlib.Tactic.Linarith
import Lean.Elab.Tactic.Omega

/-! Positive pentagonal numbers and their exact integer-square-root test. -/
namespace ProjectEulerSolutions.PentagonalDiscriminant

set_option Elab.async false

def pentagonal (n : Nat) : Nat := n * (3*n-1) / 2

def exhaustive (x : Nat) : Bool :=
  (List.range' 1 x).any fun n => decide (pentagonal n = x)

def discriminant (x : Nat) : Bool :=
  if x = 0 then false else
    decide (Nat.sqrt (24*x+1) * Nat.sqrt (24*x+1) = 24*x+1 ∧
      (1+Nat.sqrt (24*x+1))%6 = 0)

theorem twice (n : Nat) : 2*pentagonal n = n*(3*n-1) := by
  cases n with
  | zero => rfl
  | succ n =>
      have parity : n%2 = 0 ∨ n%2 = 1 := by omega
      have even : ((n+1)*(3*(n+1)-1))%2 = 0 := by
        have expression : 3*(n+1)-1 = 3*n+2 := by omega
        rcases parity with h | h <;> simp [expression, Nat.mul_mod, Nat.add_mod, h]
      have divided := Nat.mod_add_div ((n+1)*(3*(n+1)-1)) 2
      simpa only [even, Nat.zero_add, pentagonal] using divided

theorem positive (n : Nat) (positive : 0 < n) : 0 < pentagonal n := by
  have product := twice n
  have bound : 2 ≤ 3*n-1 := by omega
  nlinarith

theorem index_le (n : Nat) : n ≤ pentagonal n := by
  cases n with
  | zero => rfl
  | succ n =>
      have product := twice (n+1)
      have bound : 2 ≤ 3*(n+1)-1 := by omega
      nlinarith

theorem strictMono : StrictMono pentagonal := by
  intro a b less
  have pa := twice a
  have pb := twice b
  cases a with
  | zero => simpa [pentagonal] using positive b (by omega)
  | succ a =>
      have sa : 3*(a+1)-1 = 3*a+2 := by omega
      have sb : 3*b-1+1=3*b := by omega
      nlinarith

/-- The square-root criterion supplies exactly a positive witness in the exhaustive range. -/
theorem square_iff (x : Nat) (positiveX : 0 < x) :
    (Nat.sqrt (24*x+1)*Nat.sqrt (24*x+1) = 24*x+1 ∧ (1+Nat.sqrt (24*x+1))%6 = 0) ↔
    ∃ n, 0 < n ∧ n ≤ x ∧ pentagonal n = x := by
  constructor
  · rintro ⟨square, divisible⟩
    let root := Nat.sqrt (24*x+1)
    let n := (1+root)/6
    have rootPositive : 0 < root := by dsimp [root]; nlinarith
    have decomposed : 6*n = 1+root := by dsimp [n]; omega
    have positiveN : 0 < n := by omega
    have product : n*(3*n-1)=2*x := by
      have subtract : 3*n-1+1=3*n := by omega
      change root*root=24*x+1 at square
      nlinarith
    have equal : pentagonal n = x := by simp [pentagonal, product]
    exact ⟨n, positiveN, equal ▸ index_le n, equal⟩
  · rintro ⟨n, positiveN, _, found⟩
    have product := twice n
    rw [found] at product
    have subtract : 3*n-1+1=3*n := by omega
    have rootSubtract : 6*n-1+1=6*n := by omega
    have square : 24*x+1=(6*n-1)*(6*n-1) := by nlinarith
    rw [square, Nat.sqrt_eq]
    exact ⟨rfl, by omega⟩

theorem exhaustive_iff (x : Nat) : exhaustive x = true ↔ ∃ n, 0 < n ∧ n ≤ x ∧ pentagonal n = x := by
  simp [exhaustive, List.any_eq_true, List.mem_range']
  constructor
  · rintro ⟨i, bound, found⟩
    exact ⟨1+i, by omega, by omega, found⟩
  · rintro ⟨n, positive, bound, found⟩
    refine ⟨n-1, by omega, ?_⟩
    simpa only [show 1+(n-1)=n by omega] using found

theorem correct (x : Nat) : discriminant x = exhaustive x := by
  apply Bool.eq_iff_iff.mpr
  by_cases zero : x = 0
  · subst x
    simp [discriminant, exhaustive]
  · rw [exhaustive_iff]
    simpa only [discriminant, if_neg zero, decide_eq_true_eq] using square_iff x (by omega)

end ProjectEulerSolutions.PentagonalDiscriminant
