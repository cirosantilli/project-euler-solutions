namespace ProjectEulerSolutions.P91

partial def gcd (a b : Nat) : Nat :=
  if b == 0 then a else gcd b (a % b)

partial def countRightTriangles (n : Nat) : Nat :=
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
      loopX (x + 1) (loopY 0 acc)
  loopX 0 total


def sol : Nat :=
  countRightTriangles 50

example : countRightTriangles 2 = 14 := by
  native_decide

end ProjectEulerSolutions.P91
open ProjectEulerSolutions.P91

def main : IO Unit := do
  IO.println sol
