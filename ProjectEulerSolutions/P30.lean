import ProjectEulerStatements.P30
namespace ProjectEulerSolutions.P30

/-- Precompute the ten digit powers once for the search. -/
def digitPowers (power : Nat) : Array Nat := Array.ofFn (fun d : Fin 10 => d.val ^ power)

def digitPowerSum (powers : Array Nat) (n : Nat) : Nat :=
  (Nat.digits 10 n).foldl (fun acc d => acc + powers[d]!) 0

def solve (power : Nat) : Nat :=
  let powers := digitPowers power
  (List.range (ProjectEulerStatements.P30.limit power + 1)).foldl
    (fun total n => if n = 1 then total
      else if digitPowerSum powers n = n then total+n else total) 0
end ProjectEulerSolutions.P30
