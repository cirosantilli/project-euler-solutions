import ProjectEulerSolutions.Lib.DigitDice
import lang.py.FormalSW.Combinations

/-! The positional selection model and digit-die choices enumerate identical ordered rows. -/
namespace ProjectEulerSolutions.DigitDice
open FormalSW.PyAst

/-- Distinct positions are retained even when their labels agree. -/
theorem choices_rows (count : Nat) (pool : List α) : choices count pool = Combinations.rows count pool := by
  induction pool generalizing count with
  | nil => cases count <;> simp [choices, Combinations.rows]
  | cons first rest ih => cases count <;> simp [choices, Combinations.rows, ih]

end ProjectEulerSolutions.DigitDice
