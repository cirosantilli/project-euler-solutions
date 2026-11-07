import ProjectEulerSolutions.P25
namespace ProjectEulerSolutions.P25

theorem search_correct (digits index fuel : Nat) :
    search digits index (Nat.fib index) (Nat.fib (index+1)) fuel =
      ProjectEulerStatements.P25.firstFibWithDigits.go digits index fuel := by
  induction fuel generalizing index with
  | zero => rfl
  | succ fuel ih =>
    simp only [search, ProjectEulerStatements.P25.firstFibWithDigits.go]
    split
    · rfl
    · simpa only [show index+1+1 = index+2 by omega, Nat.fib_add_two] using ih (index+1)

theorem equiv (n : Nat) : ProjectEulerStatements.P25.naive n = solve n := by
  exact (search_correct n 0 (ProjectEulerStatements.P25.fibBound n+1)).symm
end ProjectEulerSolutions.P25
