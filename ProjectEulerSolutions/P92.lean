import ProjectEulerStatements.P92
import ProjectEulerStatements.Lib.DigitSquareDistribution
namespace ProjectEulerSolutions.P92
open ProjectEulerSolutions.DigitSquareChains

/-- Decimal prefix dynamic programming, collecting equal sum/tightness states. -/
def distribution (bound : Nat) : Prefix.Table :=
  if bound = 0 then [((0, true), 1)]
  else Prefix.transition (bound % 10) (distribution (bound / 10))
termination_by bound

def solve (limit : Int) : Nat :=
  if limit ≤ 1 then 0
  else Prefix.weighted (distribution (limit - 1).toNat)
    (fun state => if endsAt89 state.1 then 1 else 0)
end ProjectEulerSolutions.P92
