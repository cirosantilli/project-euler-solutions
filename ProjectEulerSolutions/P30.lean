import ProjectEulerStatements.P30
namespace ProjectEulerSolutions.P30

abbrev fifth : Array Nat := #[0, 1, 32, 243, 1024, 3125, 7776, 16807, 32768, 59049]

partial def digitFifthPowerSum (n : Nat) : Nat :=
  let rec loop (m acc : Nat) : Nat :=
    if m == 0 then
      acc
    else
      let d := m % 10
      loop (m / 10) (acc + fifth[d]!)
  loop n 0

partial def matchesList : List Nat :=
  let upper := 6 * fifth[9]!
  let rec loop (n : Nat) (acc : List Nat) : List Nat :=
    if n > upper then
      acc.reverse
    else
      let acc' := if n == digitFifthPowerSum n then n :: acc else acc
      loop (n + 1) acc'
  loop 2 []


example : matchesList = [4150, 4151, 54748, 92727, 93084, 194979] := by
  native_decide


partial def digitPowerSum (p n : Nat) : Nat :=
  let rec loop (m acc : Nat) : Nat :=
    if m == 0 then
      acc
    else
      let d := m % 10
      loop (m / 10) (acc + Nat.pow d p)
  if n == 0 then
    0
  else
    loop n 0

partial def sumPowerDigits (p : Nat) : Nat :=
  let upper := (p + 2) * Nat.pow 9 p
  let rec loop (n acc : Nat) : Nat :=
    if n > upper then
      acc
    else
      let acc := if n != 1 && digitPowerSum p n == n then acc + n else acc
      loop (n + 1) acc
  loop 2 0

def solve (p : Nat) :=
  if p == 5 then
    matchesList.foldl (fun acc n => acc + n) 0
  else
    sumPowerDigits p

theorem equiv (n : Nat) : ProjectEulerStatements.P30.naive n = solve n := sorry
end ProjectEulerSolutions.P30
