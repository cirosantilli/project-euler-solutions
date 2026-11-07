import ProjectEulerSolutions.Lib.TriangularDiscriminant
namespace ProjectEulerSolutions.P42

def wordValue (word : String) : Nat :=
  word.toList.foldl (fun acc c => acc + (c.toNat - 'A'.toNat + 1)) 0

def isTriangle (n : Nat) : Bool := TriangularDiscriminant.discriminant n

def stripQuotes (s : String) : String :=
  match s.toList with
  | '"' :: xs =>
      match xs.reverse with
      | '"' :: ys => String.ofList ys.reverse
      | _ => s
  | _ => s

def parseWords (s : String) : List String :=
  s.splitOn "," |>.filter (fun t => t != "") |>.map (fun t => stripQuotes t.trimAscii.toString)

def solve (words : List String) : Nat :=
  words.foldl (fun acc word => if isTriangle (wordValue word) then acc+1 else acc) 0
end ProjectEulerSolutions.P42
