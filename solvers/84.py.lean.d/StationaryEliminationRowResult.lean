import solvers.«84.py.lean.d».StationaryEliminationRow

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

def reducedRow (matrix : Matrix (Fin 120) (Fin 121) Int) (pivotRow row : Fin 120) (previous : Int) : Fin 121 → Int :=
  fun column => (matrix pivotRow pivotRow.castSucc * matrix row column - matrix row pivotRow.castSucc * matrix pivotRow column).fdiv previous

theorem rowCells_finished (matrix : Matrix (Fin 120) (Fin 121) Int) (pivotRow row : Fin 120) (previous : Int)
    (zeroRow : ∀ column : Fin 121, column.val < pivotRow.val → matrix row column = 0)
    (zeroPivot : ∀ column : Fin 121, column.val < pivotRow.val → matrix pivotRow column = 0) :
    let partialMatrix := rowCells matrix pivotRow row (matrix pivotRow pivotRow.castSucc)
      (matrix row pivotRow.castSucc) previous (pivotRow.val + 1) 121
    Function.update partialMatrix row (Function.update (partialMatrix row) pivotRow.castSucc 0) =
      Function.update matrix row (reducedRow matrix pivotRow row previous) := by
  dsimp only
  funext index column
  by_cases sameRow : index = row
  · subst index
    simp only [Function.update_self]
    by_cases sameColumn : column = pivotRow.castSucc
    · subst column
      simp [reducedRow, mul_comm]
    · rw [Function.update_of_ne sameColumn]
      have different : column.val ≠ pivotRow.val := fun equal => sameColumn (Fin.ext equal)
      by_cases earlier : column.val < pivotRow.val
      · simp [rowCells, reducedRow, zeroRow column earlier, zeroPivot column earlier, show ¬pivotRow.val + 1 ≤ column.val by omega]
      · simp [rowCells, reducedRow, show pivotRow.val + 1 ≤ column.val by omega, column.isLt]
  · simp [rowCells, sameRow]

/-- A complete source row performs its right-hand updates and explicitly clears its pivot column. -/
theorem elimination_row_result (matrix : Matrix (Fin 120) (Fin 121) Int) (pivotRow row : Fin 120)
    (different : pivotRow ≠ row) (previous : Int) (nonzero : previous ≠ 0)
    (zeroRow : ∀ column : Fin 121, column.val < pivotRow.val → matrix row column = 0)
    (zeroPivot : ∀ column : Fin 121, column.val < pivotRow.val → matrix pivotRow column = 0)
    (heap : Mutable.Heap) (root : Nat) (addresses : Fin 120 → Nat)
    (view : MutableIntegerMatrix.View matrix heap root addresses)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getRow : env "row" = some (.scalar (.int row.val)))
    (getColumn : env "column" = some (.scalar (.int pivotRow.val)))
    (getSize : env "size" = some (.scalar (.int 120)))
    (getValues : env "pivot_values" = some (.ref (addresses pivotRow)))
    (getPivot : env "pivot" = some (.scalar (.int (matrix pivotRow pivotRow.castSucc))))
    (getPrevious : env "previous" = some (.scalar (.int previous))) :
    ∃ afterEnv afterHeap,
      BlockContinues program env eliminationRowBody heap afterEnv afterHeap globals ∧
      MutableIntegerMatrix.View (Function.update matrix row (reducedRow matrix pivotRow row previous)) afterHeap root addresses ∧
      (∀ name, name ≠ "j" → name ≠ "factor" → afterEnv name = env name) ∧
      ∀ address, addresses row ≠ address → afterHeap[address]? = heap[address]? := by
  let boundEnv := setLocal env "factor" (.scalar (.int (matrix row pivotRow.castSucc)))
  have boundMatrix : boundEnv "matrix" = some (.ref root) := by simpa [boundEnv, setLocal] using getMatrix
  have boundRow : boundEnv "row" = some (.scalar (.int row.val)) := by simpa [boundEnv, setLocal] using getRow
  have boundColumn : boundEnv "column" = some (.scalar (.int pivotRow.val)) := by simpa [boundEnv, setLocal] using getColumn
  have factor := BlockContinues.assign (name := "factor")
    (view.entry_result program env globals _ _ row pivotRow.castSucc
      (view.row_result program env globals _ _ row (.lookup getMatrix) (.lookup getRow) (by intros; simp))
      (.lookup getColumn) (by intros; simp))
  obtain ⟨afterEnv, afterHeap, cells, afterView, locals, objects⟩ := cells_for matrix pivotRow row different
    (matrix pivotRow pivotRow.castSucc) (matrix row pivotRow.castSucc) previous nonzero heap root addresses view boundEnv globals
    boundMatrix boundRow boundColumn (by simpa [boundEnv, setLocal] using getSize)
    (by simpa [boundEnv, setLocal] using getValues) (by simpa [boundEnv, setLocal] using getPivot) rfl
    (by simpa [boundEnv, setLocal] using getPrevious)
  let partialMatrix := rowCells matrix pivotRow row (matrix pivotRow pivotRow.castSucc)
    (matrix row pivotRow.castSucc) previous (pivotRow.val + 1) 121
  have afterMatrix : afterEnv "matrix" = some (.ref root) := (locals _ (by decide)).trans boundMatrix
  have afterRow : afterEnv "row" = some (.scalar (.int row.val)) := (locals _ (by decide)).trans boundRow
  have afterColumn : afterEnv "column" = some (.scalar (.int pivotRow.val)) := (locals _ (by decide)).trans boundColumn
  have cleared : BlockContinues program afterEnv
      [.assign [.subscript (.subscript (.name "matrix") (.name "row")) (.name "column")] (.constant (.int 0))]
      afterHeap afterEnv (MutableIntegerMatrix.write afterHeap addresses partialMatrix row pivotRow.castSucc 0) globals := by
    apply BlockContinues.setItem (by intros; simp) Evaluates.constant
      (.cons (afterView.row_result program afterEnv globals _ _ row (.lookup afterMatrix) (.lookup afterRow) (by intros; simp))
        (.cons (.lookup afterColumn) .nil))
    exact afterView.write_operation row pivotRow.castSucc 0
  refine ⟨afterEnv, _, factor.append (cells.append cleared), ?_, ?_, ?_⟩
  · simpa only [rowCells_finished matrix pivotRow row previous zeroRow zeroPivot] using afterView.write row pivotRow.castSucc 0
  · intro name notIndex notFactor
    exact (locals name notIndex).trans (by simp [boundEnv, setLocal, notFactor])
  · intro address untouched
    exact (MutableIntegerMatrix.write_preserves row pivotRow.castSucc 0 address untouched).trans (objects address untouched)

end solvers.«84».stationary_weights
