import ProjectEulerStatements.P80

namespace ProjectEulerSolutions.P80

def sqrtFloor (n : Nat) : Nat := Nat.sqrt n

def isPerfectSquare (n : Nat) : Bool :=
  let r := sqrtFloor n
  r * r == n

def digitSumDigitsOfSqrt (n digits : Nat) : Nat :=
  ProjectEulerStatements.P80.digitSum (ProjectEulerStatements.P80.sqrtPrefix n digits)

def solve (limit digits : Nat) : Nat :=
  (List.range' 1 limit).foldl (fun total n =>
    if isPerfectSquare n then total else total + digitSumDigitsOfSqrt n digits) 0

end ProjectEulerSolutions.P80
