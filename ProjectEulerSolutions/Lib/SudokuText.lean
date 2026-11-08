import ProjectEulerSolutions.Lib.SudokuEncoding
import ProjectEulerSolutions.Lib.SudokuValues
import ProjectEulerSolutions.Lib.HeapTextLines

/-! The Sudoku text contract permits ignored non-header lines and requires unique puzzle completions. -/
namespace ProjectEulerSolutions.Sudoku.Text
open FormalSW.PyAst
set_option Elab.async false

def score (board : Board) : Int :=
  100 * Values.encode (board (0, 0)) + 10 * Values.encode (board (0, 1)) + Values.encode (board (0, 2))

/-- Records after whitespace normalization, together with their uniquely determined solutions. -/
inductive Records : List String → List Board → Prop
  | nil : Records [] []
  | skip {line : String} {rest : List String} {solved : List Board}
      (ordinary : Strings.startsWith line "Grid" = false) (tail : Records rest solved) :
      Records (line :: rest) solved
  | puzzle {header : String} {board solved : Board} {rest : List String} {solutions : List Board}
      (isHeader : Strings.startsWith header "Grid" = true)
      (completed : Completion board solved) (unique : ∀ final, Completion board final → final = solved)
      (tail : Records rest solutions) : Records (header :: Encoding.rows board ++ rest) (solved :: solutions)

def Input (text : String) (solutions : List Board) : Prop := Records (HeapTextLines.normalized text) solutions

end ProjectEulerSolutions.Sudoku.Text
