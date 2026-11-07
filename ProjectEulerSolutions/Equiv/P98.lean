import ProjectEulerSolutions.P98
namespace ProjectEulerSolutions.P98
open ProjectEulerStatements.P98 ProjectEulerSolutions.AnagramSearch

private theorem step_eq (word other : String) (root : Nat) (best : Int) :
    ProjectEulerStatements.P98.step word other root best = ProjectEulerSolutions.AnagramSearch.step word other root best := by
  simp only [ProjectEulerStatements.P98.step, ProjectEulerSolutions.AnagramSearch.step]
  cases other.toList with
  | nil => rfl
  | cons first rest => simp only [ProjectEulerSolutions.AnagramMapping.check_succeeds_iff]

private theorem roots_eq (word other : String) (first count : Nat) (best : Int) :
    roots word other first count best = scanRoots word other first count best := by
  induction count generalizing first best with
  | zero => rfl
  | succ count ih => simp only [roots, scanRoots, step_eq, ih]

private theorem pairs_eq (words : List (String × String)) (best : Int) : pairs words best = scanPairs words best := by
  induction words generalizing best with
  | nil => rfl
  | cons pair rest ih => simp only [pairs, scanPairs, roots_eq, ih]

theorem equiv (words : List String) : ProjectEulerStatements.P98.naive words = solve words :=
  pairs_eq _ 0
end ProjectEulerSolutions.P98
