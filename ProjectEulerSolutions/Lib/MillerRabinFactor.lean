import ProjectEulerSolutions.Lib.MillerRabinFast

namespace ProjectEulerSolutions.MillerRabin

private theorem naturalScan_false_of_factor (n factor : Nat) (primes : List Nat)
    (smaller : ∀ p ∈ primes, p < n) (member : factor ∈ primes) (divides : n % factor = 0) :
    Fast.naturalScan n primes = some false := by
  induction primes with
  | nil => simp at member
  | cons p rest ih =>
    have different : n ≠ p := by have := smaller p (by simp); omega
    rw [Fast.naturalScan, if_neg different]
    by_cases hit : n % p = 0
    · rw [if_pos hit]
    · rw [if_neg hit]
      apply ih
      · intro q inside
        exact smaller q (List.mem_cons_of_mem p inside)
      · rcases List.mem_cons.mp member with same | inside
        · subst factor
          exact False.elim (hit divides)
        · exact inside

/-- A divisor already tested by the actual source scan provides a short
rejection certificate. The bound excludes its small-prime equality return. -/
theorem smallFactor_rejects (n factor : Nat) (above : 37 < n)
    (member : factor ∈ smallPrimes) (divides : n % factor = 0) : Fast.model n = false := by
  have bounded : ∀ p ∈ smallPrimes, p ≤ 37 := by decide
  have scanned := naturalScan_false_of_factor n factor smallPrimes
    (fun p inside => by have := bounded p inside; omega) member divides
  simp [Fast.model, show ¬ n < 2 from by omega, scanned]

end ProjectEulerSolutions.MillerRabin
