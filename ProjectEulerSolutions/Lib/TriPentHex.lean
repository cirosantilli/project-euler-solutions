import ProjectEulerSolutions.Lib.PentagonalDiscriminant

/-! The generic polygonal-number facts used to compare the two searches for problem 45. -/
namespace ProjectEulerSolutions.TriPentHex

set_option Elab.async false

def triangular (n : Nat) : Nat := n*(n+1)/2
def hexagonal (n : Nat) : Nat := n*(2*n-1)
abbrev pentagonal := PentagonalDiscriminant.pentagonal

def isPent (x : Nat) : Bool := (List.range (x+1)).any fun n => decide (pentagonal n=x)
def isHex (x : Nat) : Bool := (List.range (x+1)).any fun n => decide (hexagonal n=x)
def common (n : Nat) : Bool := isPent (triangular n) && isHex (triangular n)
def hexTest (n : Nat) : Bool := PentagonalDiscriminant.discriminant (hexagonal n)

/-- The division by two is exact for every nonnegative triangular index. -/
theorem twice_triangular (n : Nat) : 2*triangular n=n*(n+1) := by
  have parity : n%2=0 ∨ n%2=1 := by omega
  have even : (n*(n+1))%2=0 := by
    rcases parity with h | h <;> simp [Nat.mul_mod, Nat.add_mod, h]
  have divided := Nat.mod_add_div (n*(n+1)) 2
  simpa only [even, Nat.zero_add, triangular] using divided

theorem triangular_strictMono : StrictMono triangular := by
  intro a b less
  have pa := twice_triangular a
  have pb := twice_triangular b
  nlinarith

theorem hex_positive (n : Nat) (positive : 0<n) : 0<hexagonal n := by
  have bound : 0<2*n-1 := by omega
  exact Nat.mul_pos positive bound

theorem hex_index_le (n : Nat) : n≤hexagonal n := by
  cases n with
  | zero => rfl
  | succ n =>
      have bound : 1≤2*(n+1)-1 := by omega
      dsimp [hexagonal]
      nlinarith

/-- Every positive hexagonal number occurs at the corresponding odd triangular index. -/
theorem hex_triangular (n : Nat) (positive : 0<n) : triangular (2*n-1)=hexagonal n := by
  have product := twice_triangular (2*n-1)
  have addOne : 2*n-1+1=2*n := by omega
  rw [addOne] at product
  dsimp [hexagonal]
  nlinarith

/-- The exhaustive reference includes zero, whereas the optimized predicate is only used on positive values. -/
theorem pent_positive (x : Nat) (positive : 0<x) : isPent x=PentagonalDiscriminant.discriminant x := by
  rw [PentagonalDiscriminant.correct]
  have nonzero : 0≠x := by omega
  simp only [isPent, List.range_eq_range', List.range'_succ, List.any_cons, pentagonal,
    PentagonalDiscriminant.pentagonal, Nat.zero_mul, Nat.zero_div,
    nonzero, decide_false, Bool.false_or]
  rfl

theorem hex_iff (x : Nat) : isHex x=true ↔ ∃ n, n≤x ∧ hexagonal n=x := by
  simp [isHex, List.any_eq_true, List.mem_range, Nat.lt_succ_iff]

theorem hex_witness (n : Nat) : isHex (hexagonal n)=true :=
  hex_iff _ |>.mpr ⟨n, hex_index_le n, rfl⟩

/-- A positive triangular number is hexagonal only at an odd index, with the same unique witness. -/
theorem common_hex (index : Nat) (positive : 0<index) (accepted : common index=true) :
    ∃ n, 0<n ∧ index=2*n-1 ∧ hexTest n=true := by
  obtain ⟨pent, hex⟩ := Bool.and_eq_true_iff.mp accepted
  obtain ⟨n, _, equal⟩ := (hex_iff (triangular index)).mp hex
  have positiveT : 0<triangular index := by
    have lower := triangular_strictMono positive
    simpa [triangular] using lower
  have positiveN : 0<n := by
    by_contra zero
    have nzero : n=0 := by omega
    subst n
    simp [hexagonal] at equal
    omega
  have indices : index=2*n-1 := triangular_strictMono.injective (by rw [hex_triangular n positiveN, equal])
  refine ⟨n, positiveN, indices, ?_⟩
  change PentagonalDiscriminant.discriminant (hexagonal n)=true
  rw [← pent_positive (hexagonal n) (hex_positive n positiveN), equal]
  exact pent

end ProjectEulerSolutions.TriPentHex
