import solvers.«82.py.lean.d».Allocation
import ProjectEulerSolutions.Lib.IntegerListMinimumCalls

namespace solvers.«82»
open FormalSW.PyAst HeapCalls ProjectEulerStatements.P82
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
open ProjectEulerSolutions.TriangleBufferCalls
set_option Elab.async false

abbrev outerBody : List Stmt :=
  [.assign [.name "temp"] nextComprehension, downFor, upFor, .assign [.name "dp"] (.name "temp")]
abbrev finalReturn : Stmt := .return (.builtinCall .min [.name "dp"])

def remainingColumns (rows : MatrixRows) (cursor width : Nat) (buffer : List Int) : List Int :=
  (List.range' cursor (width - cursor)).foldl (fun values index => columnStep values (column rows index)) buffer

theorem minimum_result (heap : Mutable.Heap) (env : HeapCalls.Env) (address : Nat) (buffer : List Int)
    (found : heap[address]? = some (.list (encode buffer))) (positive : 0 < buffer.length)
    (localBuffer : env "dp" = some (.ref address)) :
    Evaluates program env (.builtinCall .min [.name "dp"]) heap (number (minimum buffer)) heap := by
  cases buffer with
  | nil => simp at positive
  | cons first rest =>
      exact ProjectEulerSolutions.IntegerListMinimumCalls.minimum_evaluate program "dp" first rest heap env address localBuffer found

theorem columns_loop (heap : Mutable.Heap) (env : HeapCalls.Env) (address previousAddress cursor width : Nat)
    (rows : MatrixRows) (previous : List Int)
    (getMatrix : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (getPrevious : heap[previousAddress]? = some (.list (encode previous)))
    (shape : ∀ pair ∈ rows, pair.2.length = width) (size : previous.length = rows.length)
    (positive : 0 < rows.length) (bounded : cursor ≤ width)
    (localMatrix : env "matrix" = some (.ref address)) (localPrevious : env "dp" = some (.ref previousAddress))
    (localHeight : env "n" = some (number rows.length)) :
    ∃ extra, BlockReturns program env [.rangeStepLoop "j" cursor width 1 outerBody [], finalReturn] heap
      (number (minimum (remainingColumns rows cursor width previous))) (heap ++ extra) := by
  by_cases active : cursor < width
  · let indexed := setLocal env "j" (number cursor)
    let ready := setLocal indexed "temp" (.ref heap.length)
    let entered := addLists previous (column rows cursor)
    let updated := columnStep previous (column rows cursor)
    have bounds : ∀ pair ∈ rows, cursor < pair.2.length := by intro pair member; rw [shape pair member]; exact active
    have same : previous.length = (column rows cursor).length := by simpa [column] using size
    have allocated := next_allocation heap indexed address previousAddress cursor rows previous getMatrix getRows getPrevious
      bounds size (by simpa [indexed, setLocal] using localMatrix)
      (by simpa [indexed, setLocal] using localHeight) (by simpa [indexed, setLocal] using localPrevious) rfl
    obtain ⟨after, sweeps, kept⟩ := sweeps_result heap ready address cursor rows entered getMatrix getRows bounds
      ((addLists_length previous _ same).trans size) positive
      (by simpa [ready, indexed, setLocal] using localMatrix) rfl (by simp [ready, indexed, setLocal])
      (by simpa [ready, indexed, setLocal] using localHeight)
    let rebound := setLocal after "dp" (.ref heap.length)
    obtain ⟨more, continued⟩ := columns_loop (working heap updated) rebound address heap.length (cursor + 1) width rows updated
      (prefix_found heap updated address _ getMatrix)
      (fun pair member => prefix_found heap updated pair.1 _ (getRows pair member)) (found heap updated) shape
      ((columnStep_length previous _ same).trans size) positive (by omega)
      (by simp only [rebound, setLocal, show ("matrix" = "dp") = False by decide, if_false];
          rw [kept "matrix" (by decide) (by decide)]; simpa [ready, indexed, setLocal] using localMatrix)
      rfl
      (by simp only [rebound, setLocal, show ("n" = "dp") = False by decide, if_false];
          rw [kept "n" (by decide) (by decide)]; simpa [ready, indexed, setLocal] using localHeight)
    have next : remainingColumns rows cursor width previous = remainingColumns rows (cursor + 1) width updated := by
      unfold remainingColumns
      rw [show width - cursor = (width - (cursor + 1)) + 1 by omega, List.range'_succ, List.foldl_cons]
    refine ⟨[.list (encode updated)] ++ more, ?_⟩
    rw [next]
    apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
    apply BlockReturns.assign allocated
    have reboundBlock := BlockReturns.assign
      (Evaluates.lookup (show after "temp" = some (.ref heap.length) by rw [kept "temp" (by decide) (by decide)]; rfl)) continued
    simpa [outerBody, downFor, upFor, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
      working, List.append_assoc, entered, updated, columnStep] using sweeps.returns reboundBlock
  · have done : cursor = width := by omega
    subst cursor
    refine ⟨[], ?_⟩
    simp only [remainingColumns, Nat.sub_self, List.range'_zero, List.foldl_nil, List.append_nil]
    exact BlockReturns.range_done (by simp [Calls.rangeContinues])
      (BlockReturns.return_value (minimum_result heap env previousAddress previous getPrevious (by omega) localPrevious))
termination_by width - cursor
decreasing_by omega

end solvers.«82»
