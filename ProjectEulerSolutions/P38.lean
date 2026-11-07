import ProjectEulerSolutions.Lib.PandigitalConcat
namespace ProjectEulerSolutions.P38
open PandigitalConcat ProjectEulerStatements.P38

/-- Stop adding multiples once their concatenation reaches the target width. -/
def findCount (width seed count : Nat) : Nat → Nat
  | 0 => count
  | fuel+1 => if width ≤ numDigits (concatProduct seed count) then count
      else findCount width seed (count+1) fuel

def seedScore (width seed : Nat) : Nat :=
  let count := findCount width seed 0 (width+1)
  if accepted width seed count then concatProduct seed count else 0

def solve (width : Nat) : Nat :=
  if width < 2 ∨ 9 < width then 0
  else (List.range' 1 (seedLimit width - 1)).foldl (fun best seed => max best (seedScore width seed)) 0
end ProjectEulerSolutions.P38
