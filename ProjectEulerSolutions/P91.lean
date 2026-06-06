import ProjectEulerStatements.P91
import ProjectEulerSolutions.Termination.P91
namespace ProjectEulerSolutions.P91

def gcd (a b : Nat) : Nat :=
  if b == 0 then a else gcd b (a % b)

termination_by 0
decreasing_by all_goals exact Termination.decreases
def countRightTriangles (n : Nat) : Nat :=
  let total := n * n
  let rec loopX (x : Nat) (acc : Nat) : Nat :=
    if x > n then
      acc
    else
      let rec loopY (y : Nat) (acc : Nat) : Nat :=
        if y > n then
          acc
        else
          if x == 0 && y == 0 then
            loopY (y + 1) acc
          else if x == 0 || y == 0 then
            loopY (y + 1) (acc + n)
          else
            let g := gcd x y
            let dx := y / g
            let dy := x / g
            let k1 := Nat.min (x / dx) ((n - y) / dy)
            let k2 := Nat.min ((n - x) / dx) (y / dy)
            loopY (y + 1) (acc + k1 + k2)
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      loopX (x + 1) (loopY 0 acc)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loopX 0 total


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : countRightTriangles 2 = 14 := by
  native_decide


def solve (_n : Nat) :=
  countRightTriangles 50
end ProjectEulerSolutions.P91
