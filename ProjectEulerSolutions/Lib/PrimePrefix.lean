import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Nat.Prime.Infinite
import Mathlib.Data.Nat.Sqrt
import Mathlib.Data.List.Range
import Mathlib.Tactic.Linarith
import Lean.Elab.Tactic.Omega

/-! Complete increasing prime prefixes and the trial division used by the mutable prime-list solver. -/
namespace ProjectEulerSolutions.PrimePrefix

set_option Elab.async false

def below (bound : Nat) : List Nat := (List.range bound).filter fun p => decide (Nat.Prime p)
def through (top : Nat) : List Nat := below (top+1)
def trial (n : Nat) : List Nat → Bool
  | [] => true
  | p :: rest => if n < p*p then true else if n%p=0 then false else trial n rest

@[simp] theorem mem_below (p bound : Nat) : p ∈ below bound ↔ p < bound ∧ Nat.Prime p := by
  simp [below, List.mem_filter, List.mem_range]

@[simp] theorem mem_through (p top : Nat) : p ∈ through top ↔ p ≤ top ∧ Nat.Prime p := by
  simp [through, Nat.lt_succ_iff]

theorem ordered_below (bound : Nat) : (below bound).Pairwise (· ≤ ·) :=
  List.pairwise_le_range.filter _

theorem below_succ (bound : Nat) : below (bound+1) = below bound ++ (if Nat.Prime bound then [bound] else []) := by
  by_cases prime : Nat.Prime bound <;> simp [below, List.range_succ, List.filter_append, prime]

theorem last_through (top : Nat) (prime : Nat.Prime top) : (through top).getLast? = some top := by
  simp [through, below_succ, prime]

theorem odd_prime (top : Nat) (prime : Nat.Prime top) (large : 3≤top) : top%2=1 :=
  prime.mod_two_eq_one_iff_ne_two.mpr (by omega)

theorem next_even_composite (candidate : Nat) (odd : candidate%2=1) (large : 3≤candidate) : ¬ Nat.Prime (candidate+1) := by
  intro prime
  have parity := prime.mod_two_eq_one_iff_ne_two.mpr (by omega)
  omega

/-- Skipping the next even number does not skip an untested prime. -/
theorem below_two (candidate : Nat) (odd : candidate%2=1) (large : 3≤candidate) :
    below (candidate+2) = below candidate ++ (if Nat.Prime candidate then [candidate] else []) := by
  have composite := next_even_composite candidate odd large
  rw [show candidate+2=(candidate+1)+1 by omega, below_succ, if_neg composite, List.append_nil, below_succ]

/-- The source's early break is valid because every later divisor is at least the current one. -/
theorem trial_iff (n : Nat) (items : List Nat) (ordered : items.Pairwise (· ≤ ·)) :
    trial n items=true ↔ ∀ p ∈ items, p*p≤n → n%p≠0 := by
  induction items with
  | nil => simp [trial]
  | cons first rest ih =>
      obtain ⟨afterFirst, restOrdered⟩ := List.pairwise_cons.mp ordered
      by_cases beyond : n < first*first
      · have vacuous : ∀ p ∈ first :: rest, ¬ p*p≤n := by
          intro p member square
          rcases List.mem_cons.mp member with rfl | member
          · omega
          · have ordered := afterFirst p member
            have squares := Nat.mul_self_le_mul_self ordered
            omega
        simp only [trial, if_pos beyond, true_iff]
        intro p member square
        exact (vacuous p member square).elim
      · have square : first*first≤n := by omega
        simp only [trial, if_neg beyond, List.forall_mem_cons, square, true_implies]
        by_cases divides : n%first=0
        · simp [divides]
        · simp only [if_neg divides, ih restOrdered]
          constructor
          · intro remaining
            exact ⟨divides, remaining⟩
          · exact And.right

/-- Any increasing prime list containing all possible small prime factors gives exact primality. -/
theorem trial_correct (n : Nat) (large : 2≤n) (items : List Nat)
    (ordered : items.Pairwise (· ≤ ·)) (primeItems : ∀ p ∈ items, Nat.Prime p)
    (covered : ∀ p, Nat.Prime p → p*p≤n → p ∈ items) : trial n items=decide (Nat.Prime n) := by
  apply Bool.eq_iff_iff.mpr
  rw [trial_iff n items ordered, decide_eq_true_eq]
  constructor
  · intro checked
    by_contra composite
    have prime := Nat.minFac_prime (by omega : n≠1)
    have square := Nat.minFac_sq_le_self (by omega : 0<n) composite
    have inside := covered n.minFac prime (by simpa only [Nat.pow_two] using square)
    exact checked n.minFac inside (by simpa only [Nat.pow_two] using square) (Nat.mod_eq_zero_of_dvd (Nat.minFac_dvd n))
  · intro prime p member square divides
    have small := (primeItems p member).two_le
    obtain one | equal := (Nat.dvd_prime prime).mp (Nat.dvd_of_mod_eq_zero divides)
    · omega
    · subst p
      nlinarith

theorem trial_below (candidate : Nat) (large : 2≤candidate) :
    trial candidate (below candidate)=decide (Nat.Prime candidate) := by
  apply trial_correct candidate large _ (ordered_below candidate) (by intro p member; exact (mem_below p candidate |>.mp member).2)
  intro p prime square
  have lower := prime.two_le
  exact (mem_below p candidate).mpr ⟨by nlinarith, prime⟩

theorem trial_through (n top : Nat) (large : 2≤n) (bound : n≤top) :
    trial n (through top)=decide (Nat.Prime n) := by
  apply trial_correct n large _ (ordered_below (top+1)) (by intro p member; exact (mem_through p top |>.mp member).2)
  intro p prime square
  have lower := prime.two_le
  exact (mem_through p top).mpr ⟨by nlinarith, prime⟩

theorem initial : through 3=[2,3] := by decide +kernel

end ProjectEulerSolutions.PrimePrefix
