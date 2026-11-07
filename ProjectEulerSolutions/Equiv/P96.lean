import ProjectEulerSolutions.P96
namespace ProjectEulerSolutions.P96

theorem equiv (text : String) (input : ProjectEulerSolutions.Sudoku.Runtime.Input text) :
    ProjectEulerStatements.P96.naive text = solve text :=
  ProjectEulerSolutions.Sudoku.Runtime.run_equal text input
end ProjectEulerSolutions.P96
