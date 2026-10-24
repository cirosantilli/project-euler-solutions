import ProjectEulerSolutions.Lib.PrimeReplacementSearch
import Mathlib.Data.Nat.Find

/-! Least prime replacement families and exhaustive decimal-width windows. -/
namespace ProjectEulerSolutions.PrimeReplacementMinimum
open ProjectEulerStatements.P51
set_option Elab.async false

def accepts (target : Int) (n : Nat) : Bool := hasPrimeFamily target.toNat n
abbrev Success (target : Int) : Prop := ∃ n, accepts target n = true

noncomputable def answer (target : Int) (success : Success target) : Nat := Nat.find success

theorem answer_accepted (target : Int) (success : Success target) : accepts target (answer target success) = true :=
  Nat.find_spec success

theorem answer_minimal (target : Int) (success : Success target) (n : Nat) (accepted : accepts target n = true) :
    answer target success ≤ n := Nat.find_min' success accepted

theorem before_answer (target : Int) (success : Success target) (n : Nat) (before : n < answer target success) :
    accepts target n = false := by
  apply Bool.eq_false_iff.mpr
  intro accepted
  have least := answer_minimal target success n accepted
  omega

theorem no_success (target : Int) (absent : ¬ Success target) (n : Nat) : accepts target n = false := by
  apply Bool.eq_false_iff.mpr
  exact fun accepted => absent ⟨n, accepted⟩

theorem accepted_prime (target : Int) (n : Nat) (accepted : accepts target n = true) : Nat.Prime n := by
  simp only [accepts, hasPrimeFamily, Bool.and_eq_true, isPrime, decide_eq_true_eq] at accepted
  exact accepted.2

theorem answer_prime (target : Int) (success : Success target) : Nat.Prime (answer target success) :=
  accepted_prime target _ (answer_accepted target success)

theorem small_success (target : Int) (small : target ≤ 8) : Success target := by
  exact exists_prime_family target.toNat (by omega)

theorem large_absent (target : Int) (large : 10 < target) : ¬ Success target := by
  rintro ⟨n, accepted⟩
  have impossible := no_prime_family_above_ten target.toNat n (by omega)
  rw [accepts, impossible] at accepted
  contradiction

def lower (width : Nat) : Nat := 10 ^ (width - 1)
def upper (width : Nat) : Nat := 10 ^ width - 1
noncomputable def width (target : Int) (success : Success target) : Nat := (Nat.digits 10 (answer target success)).length

theorem width_positive (target : Int) (success : Success target) : 1 ≤ width target success := by
  have positive := (answer_prime target success).pos
  have nonempty := (Nat.digits_ne_nil_iff_ne_zero (b := 10)).mpr (by omega : answer target success ≠ 0)
  simpa only [width] using List.length_pos_iff.mpr nonempty

theorem lower_before (target : Int) (success : Success target) (first : Nat) (before : first ≤ width target success) :
    lower first ≤ answer target success := by
  have positive := width_positive target success
  have leading : 10 ^ (width target success - 1) ≤ answer target success :=
    (Nat.lt_digits_length_iff (b := 10) (k := width target success - 1) (by decide) (answer target success)).mp
      (by change width target success - 1 < width target success; omega)
  exact (Nat.pow_le_pow_right (by decide : 1 ≤ 10) (show first - 1 ≤ width target success - 1 by omega)).trans leading

theorem inside_width (target : Int) (success : Success target) : answer target success ≤ upper (width target success) := by
  have bound : answer target success < 10 ^ width target success :=
    (Nat.digits_length_le_iff (b := 10) (k := width target success) (by decide) (answer target success)).mp (Nat.le_refl _)
  unfold upper
  omega

private theorem find_least (predicate : Nat → Bool) (winner : Nat) : ∀ items : List Nat,
    items.Pairwise (· < ·) → winner ∈ items → predicate winner = true →
      (∀ p ∈ items, predicate p = true → winner ≤ p) → items.find? predicate = some winner := by
  intro items
  induction items with
  | nil => intro _ member; simp at member
  | cons head tail ih =>
      intro ordered member good least
      obtain ⟨before, rest⟩ := List.pairwise_cons.mp ordered
      cases accepted : predicate head with
      | true =>
          have lower := least head (by simp) accepted
          have upper : head ≤ winner := by
            rcases List.mem_cons.mp member with same | present
            · omega
            · exact Nat.le_of_lt (before winner present)
          have same : head = winner := by omega
          simp only [List.find?_cons, accepted]
          exact congrArg some same
      | false =>
          have present : winner ∈ tail := by
            rcases List.mem_cons.mp member with same | present
            · rw [same, accepted] at good
              contradiction
            · exact present
          have found := ih rest present good (fun p member => least p (by simp [member]))
          simpa only [List.find?_cons, accepted, Bool.false_eq_true, if_false] using found

def primes (limit : Nat) : List Nat := (List.range' 2 (limit - 1)).filter Nat.Prime

theorem mem_primes (limit p : Nat) : p ∈ primes limit ↔ Nat.Prime p ∧ p ≤ limit := by
  simp only [primes, List.mem_filter, List.mem_range', decide_eq_true_eq]
  constructor
  · rintro ⟨⟨offset, bound, rfl⟩, prime⟩
    exact ⟨prime, by omega⟩
  · rintro ⟨prime, upper⟩
    exact ⟨⟨p - 2, by have := prime.two_le; omega, by have := prime.two_le; omega⟩, prime⟩

theorem primes_pairwise (limit : Nat) : (primes limit).Pairwise (· < ·) :=
  (List.pairwise_lt_range' (s := 2) (n := limit - 1)).filter _

def window (target : Int) (digits : Nat) : Option Nat :=
  (primes (upper digits)).find? (fun n => decide (lower digits ≤ n) && accepts target n)

/-- A completed width either contains the least family or contains no qualifying prime. -/
theorem window_answer (target : Int) (success : Success target) (digits : Nat) (before : lower digits ≤ answer target success) :
    window target digits = if answer target success ≤ upper digits then some (answer target success) else none := by
  by_cases inside : answer target success ≤ upper digits
  · rw [if_pos inside]
    apply find_least _ (answer target success) (primes (upper digits)) (primes_pairwise _)
    · exact (mem_primes _ _).mpr ⟨answer_prime target success, inside⟩
    · simp only [before, decide_true, Bool.true_and]
      exact answer_accepted target success
    · intro p _ accepted
      exact answer_minimal target success p (Bool.and_eq_true_iff.mp accepted).2
  · rw [if_neg inside]
    apply List.find?_eq_none.mpr
    intro p member accepted
    have upper := (mem_primes _ p).mp member |>.2
    have least := answer_minimal target success p (Bool.and_eq_true_iff.mp accepted).2
    omega

theorem window_absent (target : Int) (absent : ¬ Success target) (digits : Nat) : window target digits = none := by
  apply List.find?_eq_none.mpr
  intro p _ accepted
  exact absent ⟨p, (Bool.and_eq_true_iff.mp accepted).2⟩

end ProjectEulerSolutions.PrimeReplacementMinimum
