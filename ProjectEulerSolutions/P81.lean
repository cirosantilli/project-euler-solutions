import ProjectEulerStatements.P81

namespace ProjectEulerSolutions.P81

def parseMatrix (text : String) : List (List Int) :=
  (text.splitOn "\n" |>.filter (fun line => line != "")).map fun line =>
    (line.splitOn ",").map fun value => (value.trimAscii.toString.toInt?).getD 0

def firstUpdate (row buffer : List Int) (index : Nat) : List Int :=
  buffer.set index (buffer.getD (index - 1) 0 + row.getD index 0)

def rowUpdate (row buffer : List Int) (index : Nat) : List Int :=
  buffer.set index (row.getD index 0 + min (buffer.getD index 0) (buffer.getD (index - 1) 0))

def firstBuffer (row : List Int) : List Int :=
  match row with
  | [] => []
  | first :: rest =>
      (List.range' 1 rest.length).foldl (firstUpdate row)
        ((List.replicate row.length 0).set 0 first)

def nextBuffer (previous row : List Int) : List Int :=
  match previous, row with
  | p :: _, x :: rest =>
      (List.range' 1 rest.length).foldl (rowUpdate row) (previous.set 0 (p + x))
  | _, _ => []

def minPathSum : List (List Int) → Int
  | [] => 0
  | row :: rows => ((rows.foldl nextBuffer (firstBuffer row)).getLast?).getD 0

def solve (matrix : List (List Int)) : Int := minPathSum matrix

end ProjectEulerSolutions.P81
