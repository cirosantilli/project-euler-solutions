import ProjectEulerSolutions.P84
namespace ProjectEulerSolutions.P84

theorem equiv (sides : Nat) (positive : 0 < sides) : ProjectEulerStatements.P84.naive sides positive = solve sides positive := by
  unfold solve
  simp only [weightArray_get]
  apply ProjectEulerSolutions.Monopoly.modal_equal sides positive
  · exact Classical.choose_spec (ProjectEulerSolutions.Monopoly.weights_exist sides positive)
  · exact ProjectEulerSolutions.Monopoly.signed_solution sides (finalState sides positive)
end ProjectEulerSolutions.P84
