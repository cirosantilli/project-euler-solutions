import solvers.«83.py.lean.d».Inputs
import ProjectEulerSolutions.Lib.MatrixSumCalls
import ProjectEulerStatements.P83

namespace solvers.«83»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RectangularMatrixCalls
set_option Elab.async false

theorem big_val.result (heap : Mutable.Heap) (address : Nat) (rows : MatrixRows)
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))) :
    ReturnsValue program "big_val" [.ref address] heap
      (.scalar (.int (ProjectEulerStatements.P83.bigVal (rows.map Prod.snd)))) heap := by
  simpa only [ProjectEulerStatements.P83.bigVal, List.foldl_map] using
    ProjectEulerSolutions.MatrixSumCalls.bound_result (program := program) rfl heap address rows found reads

end solvers.«83»
