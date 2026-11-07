import ProjectEulerSolutions.Lib.DigitFactorialBase
namespace ProjectEulerSolutions.P34
open DigitFactorialBase

/-- Compute the first digit width whose maximum factorial sum is too small. -/
def findWidth (base : Nat) (valid : 2 ≤ base) (digits : Nat)
    (before : digits ≤ stoppingWidth base valid) : Nat :=
  if done : (1 ≤ digits ∧ digits * maxFact base < base ^ (digits-1)) then digits
  else findWidth base valid (digits+1) (by
    have different : digits ≠ stoppingWidth base valid := by
      intro same
      exact done (same ▸ stopping_spec base valid)
    omega)
termination_by stoppingWidth base valid - digits
decreasing_by
  have failed : ¬ (1 ≤ digits ∧ digits * maxFact base < base ^ (digits-1)) := by assumption
  have different : digits ≠ stoppingWidth base valid := by
    intro same
    exact failed (same ▸ stopping_spec base valid)
  omega

def solve (base : Nat) (valid : 2 ≤ base) : Nat :=
  let width := findWidth base valid 1 (stopping_spec base valid).1
  let upper := (width - 1) * maxFact base
  foldValues base (List.range' base (upper + 1 - base)) 0
end ProjectEulerSolutions.P34
