import Std
import ProjectEulerStatements.P22
namespace ProjectEulerSolutions.P22

def stripQuotes (s : String) : String :=
  match s.toList with
  | '"' :: xs =>
      match xs.reverse with
      | '"' :: ys => String.ofList ys.reverse
      | _ => s
  | _ => s

def parseNames (s : String) : List String :=
  s.splitOn "," |>.filter (fun t => t != "") |>.map (fun t => stripQuotes t.trimAscii.toString)

def nameValue (name : String) : Nat :=
  name.toList.foldl (fun acc c => acc + ProjectEulerStatements.P22.letterValue c) 0

/-- Accumulate indexed name scores after sorting. -/
def scoreFrom : List String → Nat → Nat → Nat
  | [], _, total => total
  | name :: rest, index, total => scoreFrom rest (index + 1) (total + index * nameValue name)

def solve (names : List String) : Nat :=
  scoreFrom (names.mergeSort (fun a b => decide (a ≤ b))) 1 0
end ProjectEulerSolutions.P22
