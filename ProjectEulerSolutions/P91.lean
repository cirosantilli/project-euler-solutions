import ProjectEulerStatements.P91
import Mathlib.Data.Nat.GCD.Basic
namespace ProjectEulerSolutions.P91

abbrev gcd := Nat.gcd
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
      termination_by n + 1 - y
      decreasing_by all_goals omega
      loopX (x + 1) (loopY 0 acc)
  termination_by n + 1 - x
  decreasing_by omega
  loopX 0 total


def solve (n : Nat) :=
  countRightTriangles n
end ProjectEulerSolutions.P91
