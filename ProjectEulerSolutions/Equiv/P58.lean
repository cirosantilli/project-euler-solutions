import ProjectEulerSolutions.P58

namespace ProjectEulerSolutions.P58
open SpiralPrimes ProjectEulerStatements.P58

private theorem scan_eq (num den : Nat) (success : Terminates num den) (previous primes : Nat)
    (counted : primes = count previous) (before : previous < Nat.find success) :
    scan num den success previous primes counted before = side (Nat.find success) := by
  rw [scan]
  split
  · rename_i crossed
    have qualifies : 0 < previous + 1 ∧ below num den (previous + 1) := by
      refine ⟨by omega, ?_⟩
      simpa [below, count_succ, ← counted] using crossed
    have least := Nat.find_min' success qualifies
    congr 1
    omega
  · exact scan_eq num den success _ _ _ _
termination_by Nat.find success - previous

theorem equiv (num den : Nat) (success : Terminates num den) :
    ProjectEulerStatements.P58.naive num den success = solve num den success :=
  (scan_eq num den success 0 0 rfl (Nat.find_spec success).1).symm

end ProjectEulerSolutions.P58
