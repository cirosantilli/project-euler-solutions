import solvers.«96.py.lean.d».Projection
import ProjectEulerSolutions.Lib.HeapTextLines

namespace solvers.«96».parse_puzzles
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

abbrev normalization : Expr := HeapTextLines.expression "ln" (.name "text")

theorem normalization_source : ((findFunctionIn program.body "parse_puzzles").getD ([], [])).2[0]? =
    some (.assign [.name "lines"] normalization) := rfl

theorem normalize (text : String) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getText : env "text" = some (.scalar (.string text))) :
    BlockContinues program env [.assign [.name "lines"] normalization] heap
      (setLocal env "lines" (.ref (heap.length + 1)))
      (heap ++ [.list ((Strings.splitLines text).map HeapTextLines.textValue),
        .list ((HeapTextLines.normalized text).map HeapTextLines.textValue)]) globals := by
  exact BlockContinues.assign (HeapTextLines.result program "ln" (.name "text") text heap heap env globals
    (Evaluates.lookup getText) (by intro name same; cases same; simp [getText]))

end solvers.«96».parse_puzzles
