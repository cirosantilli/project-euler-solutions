import ProjectEulerSolutions.P59

namespace ProjectEulerSolutions.P59

theorem equiv (cipher : List Nat) :
    ProjectEulerStatements.P59.naive cipher = solve cipher := by
  simp only [ProjectEulerStatements.P59.naive, solve, XorDecryption.scan_streaming_eq,
    XorDecryption.best]

end ProjectEulerSolutions.P59
