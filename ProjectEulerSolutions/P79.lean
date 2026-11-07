import ProjectEulerStatements.P79

namespace ProjectEulerSolutions.P79
open ProjectEulerStatements.P79

/-- Adjacent precedence constraints suffice; select the least ready character at each step. -/
def solve (attempts : List String) : Except Error String := do
  let triples ← observations attempts
  output (ProjectEulerStatements.TopologicalOrder.order (vertices triples)
    (ProjectEulerStatements.TopologicalOrder.adjacentEdges triples))

end ProjectEulerSolutions.P79
