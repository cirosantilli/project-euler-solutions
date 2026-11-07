import ProjectEulerStatements.P93
namespace ProjectEulerSolutions.P93
open ProjectEulerSolutions.ArithmeticExpressions

/-- Each pair reduction consumes one operand; the fuel is the proved structural bound. -/
def reduceFuel : Nat → List Rat → List Rat
  | 0, _ => []
  | _ + 1, [] => []
  | _ + 1, [a] => [a]
  | fuel + 1, xs@(_ :: _ :: _) =>
      xs.permutations.flatMap fun order => match order with
        | a :: b :: rest => (operations a b).flatMap fun combined => reduceFuel fuel (combined :: rest)
        | _ => []

def score (digits : List Nat) : Nat :=
  ProjectEulerSolutions.PositiveRun.length
    (ProjectEulerStatements.P93.integers (reduceFuel digits.length (digits.map (fun n => (n : Rat)))))

def solve (maxDigit : Int) : List Nat :=
  ((ProjectEulerStatements.P93.candidates maxDigit).argmax score).getD []
end ProjectEulerSolutions.P93
