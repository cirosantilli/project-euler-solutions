import ProjectEulerStatements.P53
namespace ProjectEulerSolutions.P53

partial def countCombinatoricSelections (limitN threshold : Nat) : Nat :=
  let rec loopN (n : Nat) (total : Nat) : Nat :=
    if n > limitN then
      total
    else
      let rec loopR (r : Nat) (c : Nat) (total : Nat) : Nat :=
        if r > n / 2 then
          loopN (n + 1) total
        else
          let c := c * (n - r + 1) / r
          if c > threshold then
            let total := total + (n - 2 * r + 1)
            loopN (n + 1) total
          else
            loopR (r + 1) c total
      loopR 1 1 total
  loopN 1 0



def solveCore (limitN threshold : Nat) :=
  countCombinatoricSelections limitN threshold

def solve : Nat :=
  solveCore 100 1000000

theorem equiv :
    ProjectEulerStatements.P53.naive = solve := sorry

end ProjectEulerSolutions.P53
