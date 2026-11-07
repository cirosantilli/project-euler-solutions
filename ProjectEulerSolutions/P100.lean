import ProjectEulerStatements.P100

namespace ProjectEulerSolutions.P100

def nextSolution (u v : Int) : Int × Int := (3 * u + 4 * v, 2 * u + 3 * v)

/-- Pell-coordinate iteration used by the Python solver. -/
def search (limit u v : Int) (hu : 3 ≤ u) (hv : 1 ≤ v) : Int × Int :=
  if limit < (u + 1) / 2 then ((v + 1) / 2, (u + 1) / 2)
  else search limit (3*u+4*v) (2*u+3*v) (by omega) (by omega)
termination_by (2 * limit + 1 - u).toNat
decreasing_by omega

def blueRedForTotalLimit (limit : Int) : Int × Int × Int :=
  let p := search limit 7 5 (by omega) (by omega)
  (p.1, p.2 - p.1, p.2)

def solve (limit : Int) : Int := (blueRedForTotalLimit limit).1

end ProjectEulerSolutions.P100
