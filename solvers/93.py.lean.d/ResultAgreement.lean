import solvers.«93.py.lean.d».AllResults
import ProjectEulerSolutions.Lib.RationalResultIntegers
import data.«project-euler-statements».data.py.project_euler_statements.«p93.py.lean.d».Digits

namespace solvers.«93».all_results
open ProjectEulerSolutions.RationalPair
set_option Elab.async false

namespace Reference
abbrev integersFor := data.«project-euler-statements».data.py.project_euler_statements.p93.consecutive_for_digits.integersFor
abbrev runLength := data.«project-euler-statements».data.py.project_euler_statements.p93.consecutive_for_digits.runLength
end Reference

/-- Canonical result filtering and the complete reference search accept exactly the same integers. -/
theorem integer_results_agree (digits : List Int) (candidates : List Pair)
    (complete : ∀ pair, pair ∈ candidates ↔ Expected (integerPairs digits) pair) (n : Int) :
    n ∈ canonicalIntegerResults candidates ↔ n ∈ Reference.integersFor digits := by
  rw [expected_integer_members (integerPairs digits) candidates complete,
    data.«project-euler-statements».data.py.project_euler_statements.p93.consecutive_for_digits.mem_integersFor]
  simp [integerPairs, List.map_map, Function.comp_def, value]

theorem run_length_agrees (digits : List Int) (candidates : List Pair)
    (complete : ∀ pair, pair ∈ candidates ↔ Expected (integerPairs digits) pair) :
    ProjectEulerSolutions.PositiveRun.length (canonicalIntegerResults candidates) = Reference.runLength digits := by
  apply ProjectEulerSolutions.PositiveRun.congr
  intro n _
  exact integer_results_agree digits candidates complete n

end solvers.«93».all_results
