import solvers.«84.py.lean.d».StationaryCell

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

def rowCells (matrix : Matrix (Fin 120) (Fin 121) Int) (pivotRow row : Fin 120)
    (pivot factor previous : Int) (first stop : Nat) : Matrix (Fin 120) (Fin 121) Int :=
  fun index column => if index = row ∧ first ≤ column.val ∧ column.val < stop then
    (pivot * matrix row column - factor * matrix pivotRow column).fdiv previous else matrix index column

theorem rowCells_start (matrix : Matrix (Fin 120) (Fin 121) Int) (pivotRow row : Fin 120)
    (pivot factor previous : Int) (first : Nat) : rowCells matrix pivotRow row pivot factor previous first first = matrix := by
  funext index column
  simp only [rowCells, show ¬(index = row ∧ first ≤ column.val ∧ column.val < first) by omega, if_false]

theorem rowCells_pivot (matrix : Matrix (Fin 120) (Fin 121) Int) (pivotRow row : Fin 120)
    (different : pivotRow ≠ row) (pivot factor previous : Int) (first stop : Nat) (column : Fin 121) :
    rowCells matrix pivotRow row pivot factor previous first stop pivotRow column = matrix pivotRow column := by
  simp [rowCells, different]

theorem rowCells_current (matrix : Matrix (Fin 120) (Fin 121) Int) (pivotRow row : Fin 120)
    (pivot factor previous : Int) (first : Nat) (column : Fin 121) :
    rowCells matrix pivotRow row pivot factor previous first column.val row column = matrix row column := by
  simp [rowCells]

theorem rowCells_step (matrix : Matrix (Fin 120) (Fin 121) Int) (pivotRow row : Fin 120)
    (pivot factor previous : Int) (first : Nat) (column : Fin 121) (lower : first ≤ column.val) :
    Function.update (rowCells matrix pivotRow row pivot factor previous first column.val) row
      (Function.update (rowCells matrix pivotRow row pivot factor previous first column.val row) column
        ((pivot * matrix row column - factor * matrix pivotRow column).fdiv previous)) =
      rowCells matrix pivotRow row pivot factor previous first (column.val + 1) := by
  funext index entry
  by_cases sameRow : index = row
  · subst index
    simp only [Function.update_self]
    by_cases sameColumn : entry = column
    · subst entry
      simp [rowCells, lower]
    · rw [Function.update_of_ne sameColumn]
      have different : entry.val ≠ column.val := fun equal => sameColumn (Fin.ext equal)
      have comparison : entry.val < column.val + 1 ↔ entry.val < column.val := by omega
      simp only [rowCells, comparison]
  · simp [rowCells, sameRow]

/-- Each range iteration updates the next cell while preserving the pivot row and every other heap object. -/
theorem cells_scan (matrix : Matrix (Fin 120) (Fin 121) Int) (pivotRow row : Fin 120)
    (different : pivotRow ≠ row) (pivot factor previous : Int) (nonzero : previous ≠ 0)
    (first cursor : Nat) (lower : first ≤ cursor) (upper : cursor ≤ 121)
    (heap : Mutable.Heap) (root : Nat) (addresses : Fin 120 → Nat)
    (view : MutableIntegerMatrix.View (rowCells matrix pivotRow row pivot factor previous first cursor) heap root addresses)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getRow : env "row" = some (.scalar (.int row.val)))
    (getValues : env "pivot_values" = some (.ref (addresses pivotRow)))
    (getPivot : env "pivot" = some (.scalar (.int pivot)))
    (getFactor : env "factor" = some (.scalar (.int factor)))
    (getPrevious : env "previous" = some (.scalar (.int previous))) :
    ∃ afterEnv afterHeap,
      BlockContinues program env [.rangeStepLoop "j" cursor 121 1 cellBody []] heap afterEnv afterHeap globals ∧
      MutableIntegerMatrix.View (rowCells matrix pivotRow row pivot factor previous first 121) afterHeap root addresses ∧
      (∀ name, name ≠ "j" → afterEnv name = env name) ∧
      ∀ address, addresses row ≠ address → afterHeap[address]? = heap[address]? := by
  by_cases active : cursor < 121
  · let column : Fin 121 := ⟨cursor, active⟩
    let boundEnv := setLocal env "j" (.scalar (.int cursor))
    have boundMatrix : boundEnv "matrix" = some (.ref root) := by simpa [boundEnv, setLocal] using getMatrix
    have boundRow : boundEnv "row" = some (.scalar (.int row.val)) := by simpa [boundEnv, setLocal] using getRow
    have boundValues : boundEnv "pivot_values" = some (.ref (addresses pivotRow)) := by simpa [boundEnv, setLocal] using getValues
    have boundPivot : boundEnv "pivot" = some (.scalar (.int pivot)) := by simpa [boundEnv, setLocal] using getPivot
    have boundFactor : boundEnv "factor" = some (.scalar (.int factor)) := by simpa [boundEnv, setLocal] using getFactor
    have boundPrevious : boundEnv "previous" = some (.scalar (.int previous)) := by simpa [boundEnv, setLocal] using getPrevious
    obtain ⟨written, nextView⟩ := cell_result _ heap root addresses view pivotRow row column pivot factor previous nonzero
      boundEnv globals boundMatrix boundRow rfl boundValues boundPivot boundFactor boundPrevious
    simp only [show cursor = column.val from rfl, rowCells_current, rowCells_pivot matrix pivotRow row different] at written nextView
    rw [rowCells_step matrix pivotRow row pivot factor previous first column lower] at nextView
    obtain ⟨afterEnv, afterHeap, rest, finalView, locals, objects⟩ := cells_scan matrix pivotRow row different pivot factor previous
      nonzero first (cursor + 1) (by omega) (by omega) _ root addresses nextView boundEnv globals
      boundMatrix boundRow boundValues boundPivot boundFactor boundPrevious
    refine ⟨afterEnv, afterHeap, ?_, finalView, ?_, ?_⟩
    · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa [cellBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using written.append rest
    · intro name notIndex
      exact (locals name notIndex).trans (by simp [boundEnv, setLocal, notIndex])
    · intro address untouched
      exact (objects address untouched).trans (MutableIntegerMatrix.write_preserves row column _ address untouched)
  · have finished : cursor = 121 := by omega
    subst cursor
    exact ⟨env, heap, BlockContinues.range_done (by simp [Calls.rangeContinues]) .nil, view,
      fun _ _ => rfl, fun _ _ => rfl⟩
termination_by 121 - cursor
decreasing_by omega

theorem cells_for (matrix : Matrix (Fin 120) (Fin 121) Int) (pivotRow row : Fin 120)
    (different : pivotRow ≠ row) (pivot factor previous : Int) (nonzero : previous ≠ 0)
    (heap : Mutable.Heap) (root : Nat) (addresses : Fin 120 → Nat)
    (view : MutableIntegerMatrix.View matrix heap root addresses)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getRow : env "row" = some (.scalar (.int row.val)))
    (getColumn : env "column" = some (.scalar (.int pivotRow.val)))
    (getSize : env "size" = some (.scalar (.int 120)))
    (getValues : env "pivot_values" = some (.ref (addresses pivotRow)))
    (getPivot : env "pivot" = some (.scalar (.int pivot)))
    (getFactor : env "factor" = some (.scalar (.int factor)))
    (getPrevious : env "previous" = some (.scalar (.int previous))) :
    ∃ afterEnv afterHeap,
      BlockContinues program env ((eliminationRowBody.drop 1).take 1) heap afterEnv afterHeap globals ∧
      MutableIntegerMatrix.View (rowCells matrix pivotRow row pivot factor previous (pivotRow.val + 1) 121) afterHeap root addresses ∧
      (∀ name, name ≠ "j" → afterEnv name = env name) ∧
      ∀ address, addresses row ≠ address → afterHeap[address]? = heap[address]? := by
  obtain ⟨afterEnv, afterHeap, ran, finalView, locals, objects⟩ := cells_scan matrix pivotRow row different pivot factor previous
    nonzero (pivotRow.val + 1) (pivotRow.val + 1) (Nat.le_refl _) (by have bound := pivotRow.isLt; omega)
    heap root addresses (by simpa only [rowCells_start] using view) env globals getMatrix getRow getValues getPivot getFactor getPrevious
  refine ⟨afterEnv, afterHeap, ?_, finalView, locals, objects⟩
  apply BlockContinues.for_range (values := [.scalar (.int (pivotRow.val + 1 : Nat)), .scalar (.int 121)])
    (.cons (Evaluates.binary (.lookup getColumn) Evaluates.constant (by simp [Mutable.binary, scalarBinary]))
      (.cons (Evaluates.binary (.lookup getSize) Evaluates.constant rfl) .nil)) rfl ran

end solvers.«84».stationary_weights
