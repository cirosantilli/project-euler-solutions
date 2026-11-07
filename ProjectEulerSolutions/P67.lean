import Std
import ProjectEulerStatements.P67
import ProjectEulerSolutions.P18
namespace ProjectEulerSolutions.P67

def parseNat (s : String) : Nat :=
  s.data.foldl (fun acc c => acc * 10 + (c.toNat - '0'.toNat)) 0

def parseTriangle (text : String) : List (List Nat) :=
  let lines := text.splitOn "\n" |>.filter (fun ln => ln != "")
  lines.map (fun ln => ln.splitOn " " |>.filter (fun t => t != "") |>.map parseNat)

def toTriangle? (rows : List (List Nat)) : Option ProjectEulerStatements.P18.Triangle :=
  if h : ∀ i (hi : i < rows.length), (rows.get ⟨i, hi⟩).length = i + 1 then
    some ⟨rows, h⟩
  else
    none

def maxPathSum (triangle : List (List Nat)) : Nat :=
  ProjectEulerSolutions.P18.maxPathSum triangle


example :
    let sample := "3\n7 4\n2 4 6\n8 5 9 3"
    maxPathSum (parseTriangle sample) = 23 := by
  native_decide


def solve (triangle : ProjectEulerStatements.P18.Triangle) : Nat :=
  maxPathSum triangle.1
end ProjectEulerSolutions.P67
