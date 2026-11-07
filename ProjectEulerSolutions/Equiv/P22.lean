import ProjectEulerSolutions.P22
namespace ProjectEulerSolutions.P22

theorem scoreFrom_eq (names : List String) (index total : Nat) :
    scoreFrom names index total =
      (ProjectEulerStatements.P22.enumFrom index names).foldl
        (fun acc p => acc + ProjectEulerStatements.P22.nameScore p.1 p.2) total := by
  induction names generalizing index total with
  | nil => rfl
  | cons name rest ih =>
      simpa [scoreFrom, ProjectEulerStatements.P22.enumFrom, ProjectEulerStatements.P22.nameScore,
        nameValue, ProjectEulerStatements.P22.nameValue] using ih (index+1) (total+index*nameValue name)

theorem equiv (names : List String) : ProjectEulerStatements.P22.naive names = solve names := by
  exact (scoreFrom_eq _ 1 0).symm
end ProjectEulerSolutions.P22
