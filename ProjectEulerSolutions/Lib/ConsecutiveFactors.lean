import ProjectEulerSolutions.Lib.PrimeFactorSieve

/-! The least diagonal run and the streak invariant used by the expanding sieve search.
The existence of diagonal runs is the pre-existing statement axiom, not a new theorem. -/
namespace ProjectEulerSolutions.ConsecutiveFactors
open ProjectEulerStatements.P47

set_option Elab.async false

def good (k n : Nat) : Bool := decide (numDistinctPrimeFactors n=k)
def run (k first : Nat) : Bool := (List.range' first k).all (good k)
def least (k : Nat) : Nat := ProjectEulerStatements.P47.naive k

/-- Length of the true suffix immediately before next. -/
def streak (test : Nat → Bool) : Nat → Nat
  | 0 => 0
  | next+1 => if test next then streak test next+1 else 0

theorem streak_le (test : Nat → Bool) (next : Nat) : streak test next≤next := by
  induction next with
  | zero => rfl
  | succ next ih => simp only [streak]; split <;> omega

/-- An all-true window is exactly a sufficiently long suffix at its right boundary. -/
theorem window_iff (test : Nat → Bool) (first count : Nat) :
    (List.range' first count).all test=true ↔ count≤streak test (first+count) := by
  induction count with
  | zero => simp
  | succ count ih =>
      rw [List.range'_concat, List.all_append, Bool.and_eq_true]
      simp only [List.all_cons, List.all_nil, Bool.and_true]
      rw [show first+(count+1)=(first+count)+1 by omega, streak]
      cases value : test (first+count) <;> simp [value, ih]

theorem run_statement (k first : Nat) :
    run k first=(List.range k).all (fun j => decide (numDistinctPrimeFactors (first+j)=k)) := by
  simp [run, good, List.range'_eq_map_range, Function.comp_def]

/-- This is the exact existence axiom already used by the statement's mathematical reference. -/
theorem least_accepted (k : Nat) : run k (least k)=true := by
  rw [run_statement]
  exact Nat.find_spec (exists_consecutive_with_n_factors k)

theorem no_earlier (k first : Nat) (earlier : first<least k) : run k first=false := by
  cases accepted : run k first with
  | false => rfl
  | true =>
      rw [run_statement] at accepted
      have bound := Nat.find_min' (exists_consecutive_with_n_factors k) accepted
      dsimp [least, ProjectEulerStatements.P47.naive] at earlier
      omega

theorem least_lower (k : Nat) (positive : 0<k) : 2≤least k := by
  have accepted := least_accepted k
  have first : good k (least k)=true := by
    apply List.all_eq_true.mp accepted
    exact List.mem_range'.mpr ⟨0, positive, by simp⟩
  by_contra small
  have possibilities : least k=0 ∨ least k=1 := by omega
  rcases possibilities with zero | one
  · simp [good, zero] at first
    omega
  · simp [good, one] at first
    omega

def lastIndex (k : Nat) : Nat := least k+k-1

/-- No earlier right boundary can have accumulated k successes. -/
theorem streak_before (k next : Nat) (earlier : next<least k+k) :
    streak (good k) next<k := by
  by_contra enough
  have lower : k≤streak (good k) next := by omega
  have size := streak_le (good k) next
  have bound : k≤next := by omega
  have accepted : run k (next-k)=true := by
    rw [run, window_iff, Nat.sub_add_cancel bound]
    exact lower
  have earlierStart : next-k<least k := by omega
  rw [no_earlier k (next-k) earlierStart] at accepted
  cases accepted

/-- At the first successful right boundary the streak reaches k exactly. -/
theorem streak_winner (k : Nat) (positive : 0<k) :
    streak (good k) (lastIndex k+1)=k := by
  have start := least_lower k positive
  have enough : k≤streak (good k) (least k+k) := (window_iff _ _ _).mp (least_accepted k)
  have previous : streak (good k) (lastIndex k)<k := streak_before k (lastIndex k) (by dsimp [lastIndex]; omega)
  have boundary : lastIndex k+1=least k+k := by dsimp [lastIndex]; omega
  have oneStep : streak (good k) (lastIndex k+1)≤streak (good k) (lastIndex k)+1 := by
    rw [streak]
    split <;> omega
  rw [← boundary] at enough
  omega

theorem winner_good (k : Nat) (positive : 0<k) : good k (lastIndex k)=true := by
  have accepted := streak_winner k positive
  cases value : good k (lastIndex k) with
  | true => rfl
  | false => simp [streak, value] at accepted; omega

theorem streak_two (k : Nat) (positive : 0<k) : streak (good k) 2=0 := by
  simp [streak, good, show (0 : Nat)≠k by omega]

end ProjectEulerSolutions.ConsecutiveFactors
