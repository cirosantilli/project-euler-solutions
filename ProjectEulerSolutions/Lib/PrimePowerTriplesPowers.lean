import ProjectEulerSolutions.Lib.PrimePowerTriplesSearch
import ProjectEulerStatements.Lib.PrimePartitions
import Mathlib.Data.List.Sort

namespace ProjectEulerSolutions.PrimePowerTriples
set_option Elab.async false

def powerList (limit exponent : Nat) (primes : List Nat) : List Nat :=
  (primes.map (· ^ exponent)).filter (fun value => decide (value < limit))

theorem powerList_append (limit exponent : Nat) (left right : List Nat) :
    powerList limit exponent (left ++ right) = powerList limit exponent left ++ powerList limit exponent right := by
  simp [powerList]

theorem powerList_cons (limit exponent p : Nat) (rest : List Nat) :
    powerList limit exponent (p :: rest) =
      (if p ^ exponent < limit then [p ^ exponent] else []) ++ powerList limit exponent rest := by
  by_cases inside : p ^ exponent < limit <;> simp [powerList, inside]

theorem powerList_singleton (limit exponent p : Nat) :
    powerList limit exponent [p] = if p ^ exponent < limit then [p ^ exponent] else [] := by
  by_cases inside : p ^ exponent < limit <;> simp [powerList, inside]

theorem powerList_ordered (limit exponent : Nat) (primes : List Nat) (ordered : primes.Pairwise (· ≤ ·)) :
    (powerList limit exponent primes).Pairwise (· ≤ ·) := by
  exact (ordered.map (· ^ exponent) (fun _ _ lower => Nat.pow_le_pow_left lower exponent)).filter _

theorem primes_ordered (bound : Nat) :
    (ProjectEulerStatements.PrimePartitions.primesUpTo bound).Pairwise (· ≤ ·) :=
  List.pairwise_le_range.filter _

theorem powerList_complete (limit exponent : Nat) :
    (powerList limit exponent (ProjectEulerStatements.PrimePartitions.primesUpTo (Nat.sqrt limit + 1))).toFinset =
      powers limit exponent := by
  ext value
  simp [powerList, powers, primesThrough, ProjectEulerStatements.PrimePartitions.primesUpTo]

end ProjectEulerSolutions.PrimePowerTriples
