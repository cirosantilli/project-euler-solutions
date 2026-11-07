import lang.py.FormalSW.Calls

/-! The increasing disc-count recurrence used by both Python implementations. -/
namespace ProjectEulerSolutions.ArrangedProbability

def firstOver (limit blue total : Int) (positiveBlue : 0 < blue) (positiveTotal : 1 < total) : Int × Int :=
  if limit < total then (blue, total)
  else firstOver limit (3 * blue + 2 * total - 2) (4 * blue + 3 * total - 3) (by omega) (by omega)
termination_by (limit + 1 - total).toNat
decreasing_by omega

def answer (limit : Int) : Int := (firstOver limit 3 4 (by decide) (by decide)).1

end ProjectEulerSolutions.ArrangedProbability
