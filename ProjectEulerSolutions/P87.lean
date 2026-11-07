import ProjectEulerStatements.P87
import ProjectEulerSolutions.Lib.PrimePowerTriplesPowers

namespace ProjectEulerSolutions.P87
open ProjectEulerSolutions.PrimePowerTriples

/-- Enumerate ordered prime powers and stop each inner row at the limit. -/
def countPrimePowerTriples (limit : Nat) : Nat :=
  let primes := ProjectEulerStatements.PrimePartitions.primesUpTo (Nat.sqrt limit + 1)
  (prunedValues limit (powerList limit 2 primes) (powerList limit 3 primes)
    (powerList limit 4 primes)).toFinset.card

def solve (limit : Nat) : Nat := countPrimePowerTriples limit

theorem solve_eq (limit : Nat) : solve limit = (exhaustive limit).card := by
  unfold solve countPrimePowerTriples
  dsimp only
  rw [prunedValues_complete _ _ _ _
    (powerList_ordered _ _ _ (primes_ordered _))
    (powerList_ordered _ _ _ (primes_ordered _))
    (powerList_complete _ _) (powerList_complete _ _) (powerList_complete _ _)]

end ProjectEulerSolutions.P87
