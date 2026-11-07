import ProjectEulerSolutions.P11
namespace ProjectEulerSolutions.P11

theorem equiv (grid : List (List Nat)) (n : Nat) :
    ProjectEulerStatements.P11.naive grid n = solve grid n :=
  ProjectEulerStatements.P11.naive_eq_scanned grid n
end ProjectEulerSolutions.P11
