import ProjectEulerStatements.P31
import ProjectEulerSolutions.Lib.CoinChangeArray
namespace ProjectEulerSolutions.P31

/-- Array dynamic programming, processing each denomination once. -/
def countWays (target : Nat) (coins : List Nat) : Nat :=
  (CoinChangeArray.compute target coins)[target]!

def solve (amt : Nat) : Nat :=
  countWays amt [1, 2, 5, 10, 20, 50, 100, 200]
end ProjectEulerSolutions.P31
