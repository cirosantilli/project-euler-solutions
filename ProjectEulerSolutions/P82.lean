import ProjectEulerStatements.P82

namespace ProjectEulerSolutions.P82
open ProjectEulerStatements.P82 (getCol minimum)

def parseMatrix (text : String) : List (List Int) :=
  (text.splitOn "\n" |>.filter (fun line => line != "")).map fun line =>
    (line.splitOn ",").map fun value => (value.trimAscii.toString.toInt?).getD 0

def downUpdate (column buffer : List Int) (index : Nat) : List Int :=
  buffer.set index (min (buffer.getD index 0) (buffer.getD (index - 1) 0 + column.getD index 0))

def upUpdate (column buffer : List Int) (index : Nat) : List Int :=
  buffer.set index (min (buffer.getD index 0) (buffer.getD (index + 1) 0 + column.getD index 0))

def downBuffer (cost column : List Int) : List Int :=
  if cost.length = column.length then
    (List.range' 1 (cost.length - 1)).foldl (downUpdate column) cost
  else []

def upBuffer (cost column : List Int) : List Int :=
  if cost.length = column.length then
    (List.range' 1 (cost.length - 1)).foldl (fun buffer index => upUpdate column buffer (cost.length - 1 - index)) cost
  else []

def nextColumn (previous column : List Int) : List Int :=
  let base := List.zipWith (· + ·) previous column
  upBuffer (downBuffer base column) column

def minimalPathSumThreeWays (matrix : List (List Int)) : Int :=
  let width := (matrix.headD []).length
  if width = 0 then 0
  else minimum ((List.range' 1 (width - 1)).foldl
    (fun buffer column => nextColumn buffer (getCol matrix column)) (getCol matrix 0))

def solve (matrix : List (List Int)) : Int := minimalPathSumThreeWays matrix

end ProjectEulerSolutions.P82
