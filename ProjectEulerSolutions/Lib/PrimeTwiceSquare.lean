import ProjectEulerSolutions.Lib.PrimePrefix
import ProjectEulerSolutions.Lib.SixStepPrimality

/-! Exact prime-plus-twice-square representations and an efficient certificate predicate. -/
namespace ProjectEulerSolutions.PrimeTwiceSquare

set_option Elab.async false

def Written (n : Nat) : Prop := ∃ p k, Nat.Prime p ∧ 0<k ∧ n=p+2*k^2

def exhaustive (n : Nat) : Bool := (List.range n).any fun p => decide (Nat.Prime p) &&
  (List.range n).any fun k => decide (p+2*k^2=n)

def fast (n : Nat) : Bool := (List.range' 1 (Nat.sqrt (n/2))).any fun k =>
  decide (Nat.Prime (n-2*k^2))

def square (n p : Nat) : Bool :=
  decide (p<n ∧ (n-p)%2=0 ∧ Nat.sqrt ((n-p)/2)*Nat.sqrt ((n-p)/2)=(n-p)/2)

def fromPrimes (n top : Nat) : Bool := (PrimePrefix.through top).any (square n)

def oddComposite (n : Nat) : Bool := decide (n%2=1 ∧ ¬ Nat.Prime n ∧ 1<n)
def counterexample (n : Nat) : Bool := oddComposite n && !fast n

/-- The same square scan with a primality test suitable for finite kernel certificates. -/
def certifiedFast (n : Nat) : Bool := (List.range' 1 (Nat.sqrt (n/2))).any fun k =>
  SixStepPrimality.isPrime (n-2*k^2)

theorem certifiedFast_correct (n : Nat) : certifiedFast n=fast n := by
  simp only [certifiedFast, fast, SixStepPrimality.isPrime_correct]

/-- The original nested ranges contain exactly the positive-square representations. -/
theorem exhaustive_iff (n : Nat) : exhaustive n=true ↔ Written n := by
  simp only [exhaustive, List.any_eq_true, List.mem_range, Bool.and_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨p, bound, prime, k, _, equation⟩
    have positive : 0<k := by
      by_contra missing
      have zero : k=0 := by omega
      subst k
      simp at equation
      omega
    exact ⟨p, k, prime, positive, equation.symm⟩
  · rintro ⟨p, k, prime, positive, equation⟩
    have lower := prime.two_le
    have positiveSquare : 0<k^2 := Nat.pow_pos positive
    have bounded : k<n := by nlinarith [Nat.le_mul_self k]
    exact ⟨p, by nlinarith, prime, k, bounded, equation.symm⟩

/-- The short square scan used by certificates is equivalent to the original quadratic search. -/
theorem fast_iff (n : Nat) : fast n=true ↔ Written n := by
  simp only [fast, List.any_eq_true, List.mem_range', decide_eq_true_eq]
  constructor
  · rintro ⟨k, ⟨offset, bound, equal⟩, prime⟩
    have positive : 0<k := by omega
    have rootBound : k≤Nat.sqrt (n/2) := by omega
    have squareBound := Nat.le_sqrt'.mp rootBound
    have subtraction : 2*k^2≤n := by omega
    exact ⟨n-2*k^2, k, prime, positive, by omega⟩
  · rintro ⟨p, k, prime, positive, equation⟩
    have bound : k≤Nat.sqrt (n/2) := by apply Nat.le_sqrt'.mpr; omega
    have recovered : n-2*k^2=p := by omega
    exact ⟨k, ⟨k-1, by omega, by omega⟩, by simpa only [recovered] using prime⟩

theorem fast_correct (n : Nat) : fast n=exhaustive n := by
  apply Bool.eq_iff_iff.mpr
  rw [fast_iff, exhaustive_iff]

/-- The remainder check with an exact integer square root identifies the same square witness. -/
theorem square_iff (n p : Nat) (prime : Nat.Prime p) :
    square n p=true ↔ ∃ k, 0<k ∧ n=p+2*k^2 := by
  simp only [square, decide_eq_true_eq]
  constructor
  · rintro ⟨less, even, square⟩
    let k := Nat.sqrt ((n-p)/2)
    have decomposition := Nat.mod_add_div (n-p) 2
    simp only [even, Nat.zero_add] at decomposition
    have remainderPositive : 0<n-p := Nat.sub_pos_of_lt less
    change k*k=(n-p)/2 at square
    have positive : 0<k := by
      nlinarith
    exact ⟨k, positive, by nlinarith [Nat.sub_add_cancel (Nat.le_of_lt less)]⟩
  · rintro ⟨k, positive, equation⟩
    have lower := prime.two_le
    have remainder : n-p=2*k^2 := by omega
    have positiveSquare : 0<k^2 := Nat.pow_pos positive
    refine ⟨by nlinarith, ?_, ?_⟩
    · omega
    · rw [remainder]
      simp [Nat.sqrt_eq, Nat.pow_two]

/-- Once the shared prefix reaches n, searching its lists is a complete search over all possible primes. -/
theorem fromPrimes_iff (n top : Nat) (bound : n≤top) : fromPrimes n top=true ↔ Written n := by
  simp only [fromPrimes, List.any_eq_true]
  constructor
  · rintro ⟨p, member, accepted⟩
    have prime := (PrimePrefix.mem_through p top).mp member |>.2
    obtain ⟨k, positive, equation⟩ := (square_iff n p prime).mp accepted
    exact ⟨p, k, prime, positive, equation⟩
  · rintro ⟨p, k, prime, positive, equation⟩
    have positiveSquare : 0<k^2 := Nat.pow_pos positive
    refine ⟨p, (PrimePrefix.mem_through p top).mpr ⟨by nlinarith, prime⟩, ?_⟩
    exact (square_iff n p prime).mpr ⟨k, positive, equation⟩

theorem fromPrimes_correct (n top : Nat) (bound : n≤top) : fromPrimes n top=fast n := by
  apply Bool.eq_iff_iff.mpr
  rw [fromPrimes_iff n top bound, fast_iff]

end ProjectEulerSolutions.PrimeTwiceSquare
