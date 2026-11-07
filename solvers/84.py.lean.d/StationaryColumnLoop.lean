import solvers.«84.py.lean.d».StationaryColumnRows

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

theorem column_scan (matrix : Matrix (Fin 120) (Fin 121) Int) (pivot : Fin 120) (previous : Int) (nonzero : previous ≠ 0)
    (eliminated : ∀ row column, column.val < pivot.val → column.val < row.val → matrix row column = 0)
    (cursor : Nat) (lower : pivot.val < cursor) (upper : cursor ≤ 120)
    (heap : Mutable.Heap) (root : Nat) (addresses : Fin 120 → Nat)
    (view : MutableIntegerMatrix.View (columnRows matrix pivot previous cursor) heap root addresses)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getColumn : env "column" = some (.scalar (.int pivot.val)))
    (getSize : env "size" = some (.scalar (.int 120)))
    (getValues : env "pivot_values" = some (.ref (addresses pivot)))
    (getPivot : env "pivot" = some (.scalar (.int (matrix pivot pivot.castSucc))))
    (getPrevious : env "previous" = some (.scalar (.int previous))) :
    ∃ afterEnv afterHeap,
      BlockContinues program env [.rangeStepLoop "row" cursor 120 1 eliminationRowBody []] heap afterEnv afterHeap globals ∧
      MutableIntegerMatrix.View (columnRows matrix pivot previous 120) afterHeap root addresses ∧
      (∀ name, name ≠ "row" → name ≠ "j" → name ≠ "factor" → afterEnv name = env name) ∧
      ∀ address, (∀ row, addresses row ≠ address) → afterHeap[address]? = heap[address]? := by
  by_cases active : cursor < 120
  · let row : Fin 120 := ⟨cursor, active⟩
    let boundEnv := setLocal env "row" (.scalar (.int cursor))
    have different : pivot ≠ row := by intro equal; have same := congrArg Fin.val equal; dsimp [row] at same; omega
    have boundMatrix : boundEnv "matrix" = some (.ref root) := by simpa [boundEnv, setLocal] using getMatrix
    have boundColumn : boundEnv "column" = some (.scalar (.int pivot.val)) := by simpa [boundEnv, setLocal] using getColumn
    have boundSize : boundEnv "size" = some (.scalar (.int 120)) := by simpa [boundEnv, setLocal] using getSize
    have boundValues : boundEnv "pivot_values" = some (.ref (addresses pivot)) := by simpa [boundEnv, setLocal] using getValues
    have boundPivot : boundEnv "pivot" = some (.scalar (.int (matrix pivot pivot.castSucc))) := by simpa [boundEnv, setLocal] using getPivot
    have boundPrevious : boundEnv "previous" = some (.scalar (.int previous)) := by simpa [boundEnv, setLocal] using getPrevious
    obtain ⟨rowEnv, rowHeap, ran, nextView, rowLocals, rowObjects⟩ := elimination_row_result
      (columnRows matrix pivot previous cursor) pivot row different previous nonzero
      (by intro column before; simpa only [show cursor = row.val from rfl, columnRows_current] using eliminated row column before (by dsimp [row]; omega))
      (by intro column before; simpa only [columnRows_pivot] using eliminated pivot column before before)
      heap root addresses view boundEnv globals boundMatrix rfl boundColumn boundSize boundValues
      (by simpa only [columnRows_pivot] using boundPivot) boundPrevious
    rw [show cursor = row.val from rfl, columnRows_reduction, columnRows_step matrix pivot row previous lower] at nextView
    obtain ⟨afterEnv, afterHeap, rest, finalView, locals, objects⟩ := column_scan matrix pivot previous nonzero eliminated
      (cursor + 1) (by omega) (by omega) rowHeap root addresses nextView rowEnv globals
      ((rowLocals _ (by decide) (by decide)).trans boundMatrix)
      ((rowLocals _ (by decide) (by decide)).trans boundColumn)
      ((rowLocals _ (by decide) (by decide)).trans boundSize)
      ((rowLocals _ (by decide) (by decide)).trans boundValues)
      ((rowLocals _ (by decide) (by decide)).trans boundPivot)
      ((rowLocals _ (by decide) (by decide)).trans boundPrevious)
    refine ⟨afterEnv, afterHeap, ?_, finalView, ?_, ?_⟩
    · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa only [Calls.loopContinuation, show Calls.loopControl eliminationRowBody = false from rfl,
        Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one] using ran.append rest
    · intro name notRow notIndex notFactor
      exact (locals name notRow notIndex notFactor).trans
        ((rowLocals name notIndex notFactor).trans (by simp [boundEnv, setLocal, notRow]))
    · intro address untouched
      exact (objects address untouched).trans (rowObjects address (untouched row))
  · have finished : cursor = 120 := by omega
    subst cursor
    exact ⟨env, heap, BlockContinues.range_done (by simp [Calls.rangeContinues]) .nil, view,
      fun _ _ _ _ => rfl, fun _ _ => rfl⟩
termination_by 120 - cursor
decreasing_by omega

/-- Execute every row below the installed pivot, producing exactly the next Bareiss state. -/
theorem column_result {sides k : Nat} (state : Bareiss.State (Monopoly.augmentedMatrix sides) k)
    (unfinished : k < 120) (nonzero : state.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩ ≠ 0)
    (heap : Mutable.Heap) (root : Nat) (addresses : Fin 120 → Nat)
    (view : MutableIntegerMatrix.View state.matrix heap root addresses)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getColumn : env "column" = some (.scalar (.int k)))
    (getSize : env "size" = some (.scalar (.int 120)))
    (getValues : env "pivot_values" = some (.ref (addresses ⟨k, unfinished⟩)))
    (getPivot : env "pivot" = some (.scalar (.int (state.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩))))
    (getPrevious : env "previous" = some (.scalar (.int state.previous))) :
    ∃ afterEnv afterHeap,
      BlockContinues program env ((eliminationBody.drop 6).take 1) heap afterEnv afterHeap globals ∧
      MutableIntegerMatrix.View (state.advance unfinished nonzero).matrix afterHeap root addresses ∧
      (∀ name, name ≠ "row" → name ≠ "j" → name ≠ "factor" → afterEnv name = env name) ∧
      ∀ address, (∀ row, addresses row ≠ address) → afterHeap[address]? = heap[address]? := by
  obtain ⟨afterEnv, afterHeap, ran, finalView, locals, objects⟩ := column_scan state.matrix ⟨k, unfinished⟩ state.previous
    state.previousNonzero state.eliminated (k + 1) (Nat.lt_succ_self k) (Nat.succ_le_of_lt unfinished) heap root addresses
    (by simpa only [columnRows_start] using view) env globals getMatrix getColumn getSize getValues getPivot getPrevious
  refine ⟨afterEnv, afterHeap, ?_, ?_, locals, objects⟩
  · apply BlockContinues.for_range (values := [.scalar (.int (k + 1 : Nat)), .scalar (.int 120)])
      (.cons (Evaluates.binary (.lookup getColumn) Evaluates.constant (by simp [Mutable.binary, scalarBinary]))
        (.cons (.lookup getSize) .nil)) rfl ran
  · simpa only [columnRows_finished state unfinished nonzero] using finalView

end solvers.«84».stationary_weights
