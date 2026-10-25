import ProjectEulerStatements.P65
import ProjectEulerStatements.Lib.DecimalDigits

namespace ProjectEulerSolutions.EConvergents
set_option Elab.async false

abbrev coefficient := ProjectEulerStatements.P65.coeffE
abbrev coefficients := ProjectEulerStatements.P65.coeffs

def advance (state : Nat × Nat) (value : Nat) : Nat × Nat :=
  (state.2, value * state.2 + state.1)

def run (values : List Nat) (older numerator : Nat) : Nat :=
  (values.foldl advance (older, numerator)).2

def numerator (n : Nat) : Nat := run (coefficients n) 0 1
def answer (n : Nat) : Nat := (Nat.digits 10 (numerator n)).sum

theorem run_nil (older numerator : Nat) : run [] older numerator = numerator := rfl
theorem run_cons (value : Nat) (rest : List Nat) (older numerator : Nat) :
    run (value :: rest) older numerator = run rest numerator (value * numerator + older) := rfl

theorem reference_run (values : List Nat) (older numerator : Nat) :
    ProjectEulerStatements.P65.convergentNum.go values older numerator = run values older numerator := by
  induction values generalizing older numerator with
  | nil => rfl
  | cons value rest ih => exact ih numerator (value * numerator + older)

theorem answer_eq_naive (n : Nat) : answer n = ProjectEulerStatements.P65.naive n := by
  unfold answer numerator ProjectEulerStatements.P65.naive ProjectEulerStatements.P65.digitSum
    ProjectEulerStatements.P65.convergentNum
  rw [reference_run]

theorem run_indices (indices : List Nat) (older numerator : Nat) :
    run (indices.map coefficient) older numerator =
      ((indices.foldl (fun state index => advance state (coefficient index)) (older, numerator)).2) := by
  rw [run, List.foldl_map]

end ProjectEulerSolutions.EConvergents
