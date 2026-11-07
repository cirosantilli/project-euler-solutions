import ProjectEulerStatements.P96
namespace ProjectEulerSolutions.P96

/-- Backtrack with cells ordered by their candidate counts. -/
def solve (text : String) : Option Nat := ProjectEulerSolutions.Sudoku.Runtime.run
  ProjectEulerSolutions.Sudoku.Runtime.constrainedOrder text
end ProjectEulerSolutions.P96
