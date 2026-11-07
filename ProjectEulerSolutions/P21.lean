import ProjectEulerSolutions.Lib.P21DivisorSieve
import ProjectEulerStatements.Lib.DivisorSumFactors
namespace ProjectEulerSolutions.P21
open ProjectEulerStatements.DivisorSumFactors ProjectEulerStatements.FactorDivisors

theorem strip_smaller (n : Nat) (large : 1 < n) : (strip n n.minFac).1 < n := by
  have prime := Nat.minFac_prime (by omega : n ≠ 1)
  have bound := strip_le n n.minFac (by omega) prime
  have excluded := (strip_spec n n.minFac (by omega) prime).2.1
  have different : (strip n n.minFac).1 ≠ n := by
    intro same
    rw [same] at excluded
    exact excluded (Nat.minFac_dvd n)
  omega

/-- Factor the uncached partner and multiply the prime-power geometric sums. -/
def sigmaFast (n : Nat) : Nat :=
  if n ≤ 1 then n else
    let factor := n.minFac
    let removed := strip n factor
    geometric factor removed.2 * sigmaFast removed.1
termination_by n
decreasing_by exact strip_smaller n (by omega)


/-- Reuse the sieve within its allocated interval; compute a divisor sum for larger partners. -/
def cachedSum (sums : Array Nat) (n : Nat) : Nat :=
  if n < sums.size then sums[n]! else sigmaFast n - n

def solve (limit : Nat) : Nat :=
  let sums := P21DivisorSieve.compute (limit-1)
  ((List.range' 1 (limit-1)).filter fun n =>
    let partner := cachedSum sums n
    decide (partner ≠ n ∧ cachedSum sums partner = n)).sum
end ProjectEulerSolutions.P21
