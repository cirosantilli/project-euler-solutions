import ProjectEulerSolutions.P51

namespace ProjectEulerSolutions.P51

theorem equiv (target : Nat) (success : ∃ n, ProjectEulerStatements.P51.hasPrimeFamily target n = true) :
    ProjectEulerStatements.P51.naive target = solve target success := by
  rw [ProjectEulerStatements.P51.naive, dif_pos success]
  unfold solve
  rw [ProjectEulerStatements.FirstWitness.first_eq_find]
  simp only [accepts_eq]

end ProjectEulerSolutions.P51
