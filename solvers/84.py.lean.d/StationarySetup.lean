import solvers.«84.py.lean.d».StationaryMatrixView

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

def setupEnv (env : HeapCalls.Env) (heap : Mutable.Heap) : HeapCalls.Env :=
  setLocal (setLocal env "size" (.scalar (.int 120))) "matrix" (.ref heap.length)

theorem setup_full_source : (executableBody definition.2).take 3 =
    [.assign [.name "size"] (.builtinCall .len [.name "transitions"]), .assign [.name "matrix"] matrixExpression,
      .expr (.call (.attribute (.name "matrix") "append") [normalizationExpression])] := rfl

/-- Execute the complete matrix construction, including its final normalization equation. -/
theorem setup_result (sides : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (value : Mutable.Value) (view : transitions.View sides heap value)
    (getTransitions : env "transitions" = some value)
    (getDenominator : env "denominator" = some (.scalar (.int (256 * (sides : Int) * sides)))) :
    BlockContinues program env ((executableBody definition.2).take 3) heap (setupEnv env heap) (preparedHeap heap sides) globals ∧
      MatrixView sides (preparedHeap heap sides) heap.length ∧ ∃ extra, preparedHeap heap sides = heap ++ extra := by
  refine ⟨?_, prepared_view heap sides, prepared_prefix heap sides⟩
  rw [setup_full_source]
  have size := BlockContinues.assign (name := "size") (size_result sides heap env globals value view getTransitions)
  have matrix := BlockContinues.assign (name := "matrix")
    (matrix_result sides heap (setLocal env "size" (.scalar (.int 120))) globals value view
      (by simpa [setLocal] using getTransitions) (by simpa [setLocal] using getDenominator) (by simp [setLocal]))
  have normalized := append_normalization sides heap (setupEnv env heap) globals
    (by simp [setupEnv, setLocal]) (by simp [setupEnv, setLocal])
  exact size.append (matrix.append normalized)

end solvers.«84».stationary_weights
