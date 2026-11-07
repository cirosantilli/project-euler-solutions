import ProjectEulerStatements.P86
namespace ProjectEulerSolutions.P86
open ProjectEulerSolutions.CuboidRoute

def countNewSolutions (c : Nat) : Nat := groupedLayer c

def countSolutionsUpTo (m : Nat) : Nat := groupedTotal m

/-- Maintain the running count while enumerating each new largest side. -/
def search (target : Int) (c count : Nat) (correct : count = total c) : Nat :=
  if active : (count : Int) ≤ target then
    search target (c + 1) (count + countNewSolutions (c + 1))
      (by rw [correct, total, countNewSolutions, layer_equivalence])
  else c
termination_by searchBound target - c
decreasing_by
  have := active_below_bound target c (by simpa [correct] using active)
  omega

def solve (target : Int) : Nat := search target 0 0 rfl
end ProjectEulerSolutions.P86
