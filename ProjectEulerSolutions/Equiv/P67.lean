import ProjectEulerSolutions.P67
import ProjectEulerSolutions.Equiv.P18

namespace ProjectEulerSolutions.P67

theorem equiv (triangle : ProjectEulerStatements.P18.Triangle) :
    ProjectEulerStatements.P67.naive triangle = solve triangle :=
  ProjectEulerSolutions.P18.equiv triangle

end ProjectEulerSolutions.P67
