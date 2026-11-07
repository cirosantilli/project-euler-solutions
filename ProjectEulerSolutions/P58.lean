import ProjectEulerStatements.P58

namespace ProjectEulerSolutions.P58
open SpiralPrimes ProjectEulerStatements.P58

def scan (num den : Nat) (success : Terminates num den) (previous primes : Nat)
    (counted : primes = count previous) (before : previous < Nat.find success) : Nat :=
  let layer := previous + 1
  let updated := primes + row layer
  if crossed : den * updated < num * total layer then side layer
  else scan num den success layer updated (by simp only [updated, layer, count_succ, counted]) (by
    have first := Nat.find_spec success
    have notLast : layer ≠ Nat.find success := by
      intro same
      apply crossed
      have crossing := first.2
      change den * count (Nat.find success) < num * total (Nat.find success) at crossing
      rw [← same] at crossing
      simpa only [updated, layer, count_succ, counted] using crossing
    omega)
termination_by Nat.find success - previous

def solve (num den : Nat) (success : Terminates num den) : Nat :=
  scan num den success 0 0 rfl (Nat.find_spec success).1

end ProjectEulerSolutions.P58
