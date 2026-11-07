import ProjectEulerSolutions.P34
namespace ProjectEulerSolutions.P34
open DigitFactorialBase

theorem findWidth_correct (base : Nat) (valid : 2 ≤ base) (digits : Nat)
    (before : digits ≤ stoppingWidth base valid) :
    findWidth base valid digits before = stoppingWidth base valid := by
  rw [findWidth]
  split
  · rename_i done
    have least : stoppingWidth base valid ≤ digits := Nat.find_min' (exists_bad_width base valid) done
    omega
  · exact findWidth_correct base valid (digits+1) _
termination_by stoppingWidth base valid - digits
decreasing_by
  have failed : ¬ (1 ≤ digits ∧ digits * maxFact base < base ^ (digits-1)) := by assumption
  have different : digits ≠ stoppingWidth base valid := by
    intro same
    exact failed (same ▸ stopping_spec base valid)
  omega

/-- `answer` is the complete set of base-dependent digit-factorial fixed points. -/
theorem equiv (base : Nat) (valid : 2 ≤ base) : answer base valid = solve base valid := by
  unfold solve
  rw [findWidth_correct]
  exact (range_answer base valid).symm
end ProjectEulerSolutions.P34
