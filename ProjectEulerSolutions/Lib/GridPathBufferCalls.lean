import ProjectEulerSolutions.Lib.TriangleBufferCalls
import ProjectEulerStatements.Lib.GridPathRows

namespace ProjectEulerSolutions.GridPathBufferCalls
open FormalSW.PyAst HeapCalls
open ProjectEulerStatements.P81
open ProjectEulerSolutions.TriangleBufferCalls
set_option Elab.async false

private theorem getD_some (values : List Int) (index : Nat) (inside : index < values.length) :
    values[index]? = some (values.getD index 0) := by
  simp [List.getD, List.getElem?_eq_getElem inside]

private theorem integer_minimum (a b : Int) (heap : Mutable.Heap) :
    HeapCalls.primitive (.builtin .min) [number a, number b] heap =
      some (.ok (number (min a b)), heap) := by
  by_cases smaller : b < a
  · simp [HeapCalls.primitive, Calls.positionalMinimum, Calls.minimumFrom, scalarCompare,
      evalCmpOp, HeapCalls.scalarResult, List.mapM, List.mapM.loop, smaller, min_eq_right (le_of_lt smaller)]
  · simp [HeapCalls.primitive, Calls.positionalMinimum, Calls.minimumFrom, scalarCompare,
      evalCmpOp, HeapCalls.scalarResult, List.mapM, List.mapM.loop, smaller, min_eq_left (le_of_not_gt smaller)]

def rowElement : Expr := .binOp (.subscript (.name "row") (.name "j")) .add
  (.builtinCall .min [.subscript (.name "dp") (.name "j"),
    .subscript (.name "dp") (.binOp (.name "j") .sub (.constant (.int 1)))])
def rowBody : List Stmt := [.assign [.subscript (.name "dp") (.name "j")] rowElement]

theorem row_element_result (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (address cursor : Nat) (previous row : List Int)
    (readRow : base[address]? = some (.list (encode row)))
    (same : previous.length = row.length) (positive : 0 < cursor) (inside : cursor < row.length)
    (localRow : env "row" = some (.ref address))
    (localBuffer : env "dp" = some (.ref base.length))
    (localCursor : env "j" = some (number cursor)) :
    Evaluates program env rowElement (working base (partialBuffer (rowStep previous row) previous cursor))
      (number ((rowStep previous row).getD cursor 0))
      (working base (partialBuffer (rowStep previous row) previous cursor)) := by
  let target := rowStep previous row
  let state := partialBuffer target previous cursor
  have targetLength : target.length = previous.length := by simp [target, same]
  have after : state[cursor]? = some (previous.getD cursor 0) := by
    rw [partialBuffer_suffix target previous cursor cursor (by omega) (by omega)]
    exact getD_some _ _ (by omega)
  have before : state[cursor - 1]? = some (target.getD (cursor - 1) 0) := by
    rw [partialBuffer_prefix target previous cursor (cursor - 1) (by omega) (by omega)]
    exact getD_some _ _ (by omega)
  have previousIndex : Evaluates program env (.binOp (.name "j") .sub (.constant (.int 1)))
      (working base state) (number (cursor - 1 : Nat)) (working base state) := by
    exact Evaluates.binary (Evaluates.lookup localCursor) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, number, Int.ofNat_sub (by omega : 1 ≤ cursor)])
  have current := index_evaluates program _ env "dp" base.length cursor state _ (.name "j")
    localBuffer (Evaluates.lookup localCursor) (by intros a b c impossible; cases impossible) (found base state) after
  have left := index_evaluates program _ env "dp" base.length (cursor - 1) state _ _
    localBuffer previousIndex (by intros a b c impossible; cases impossible) (found base state) before
  have minimum : Evaluates program env
      (.builtinCall .min [.subscript (.name "dp") (.name "j"),
        .subscript (.name "dp") (.binOp (.name "j") .sub (.constant (.int 1)))])
      (working base state) (number (min (previous.getD cursor 0) (target.getD (cursor - 1) 0)))
      (working base state) := by
    apply Evaluates.builtin (.cons current (.cons left .nil))
    exact integer_minimum _ _ _
  have recurrence : target.getD cursor 0 = row.getD cursor 0 +
      min (previous.getD cursor 0) (target.getD (cursor - 1) 0) := by
    simpa only [Nat.sub_add_cancel (by omega : 1 ≤ cursor)] using
      rowStep_recurrence previous row (cursor - 1) (by omega) same
  rw [recurrence]
  exact Evaluates.binary
    (index_evaluates program _ env "row" address cursor row _ (.name "j") localRow
      (Evaluates.lookup localCursor) (by intros a b c impossible; cases impossible)
      (prefix_found base state address _ readRow) (getD_some row cursor inside)) minimum rfl

theorem row_update_result (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (address cursor : Nat) (previous row : List Int)
    (readRow : base[address]? = some (.list (encode row)))
    (same : previous.length = row.length) (positive : 0 < cursor) (inside : cursor < row.length)
    (localRow : env "row" = some (.ref address))
    (localBuffer : env "dp" = some (.ref base.length))
    (localCursor : env "j" = some (number cursor)) :
    BlockContinues program env rowBody (working base (partialBuffer (rowStep previous row) previous cursor)) env
      (working base (partialBuffer (rowStep previous row) previous (cursor + 1))) := by
  have targetLength : (rowStep previous row).length = previous.length := by simp [same]
  apply BlockContinues.setItem (by intros a b c impossible; cases impossible)
    (row_element_result program base env address cursor previous row readRow same positive inside localRow localBuffer localCursor)
    (.cons (Evaluates.lookup localBuffer) (.cons (Evaluates.lookup localCursor) .nil))
  have updated := set_item base (partialBuffer (rowStep previous row) previous cursor) cursor
    ((rowStep previous row).getD cursor 0) (by rw [partialBuffer_length _ _ _ targetLength]; omega)
  have equal : (rowStep previous row).getD cursor 0 = (rowStep previous row)[cursor]'(by omega) := by
    simp [List.getD, List.getElem?_eq_getElem (by omega : cursor < (rowStep previous row).length)]
  rw [equal, partialBuffer_step (rowStep previous row) previous cursor targetLength (by omega)] at updated
  simpa only [equal] using updated

theorem row_loop (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (address cursor : Nat) (previous row : List Int)
    (readRow : base[address]? = some (.list (encode row)))
    (same : previous.length = row.length) (positive : 0 < cursor) (bounded : cursor ≤ row.length)
    (localRow : env "row" = some (.ref address)) (localBuffer : env "dp" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "j" cursor row.length 1 rowBody []]
      (working base (partialBuffer (rowStep previous row) previous cursor)) finalEnv
      (working base (rowStep previous row)) ∧
      ∀ name, name ≠ "j" → finalEnv name = env name := by
  by_cases active : cursor < row.length
  · let indexed := setLocal env "j" (number cursor)
    have step := row_update_result program base indexed address cursor previous row readRow same positive active
      (by simpa [indexed, setLocal] using localRow) (by simpa [indexed, setLocal] using localBuffer) rfl
    obtain ⟨finalEnv, rest, kept⟩ := row_loop program base indexed address (cursor + 1) previous row readRow same
      (by omega) (by omega) (by simpa [indexed, setLocal] using localRow) (by simpa [indexed, setLocal] using localBuffer)
    refine ⟨finalEnv, ?_, ?_⟩
    · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa [rowBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using step.append rest
    · intro name different
      rw [kept name different]
      simp [indexed, setLocal, different]
  · have done : cursor = row.length := by omega
    subst cursor
    refine ⟨env, ?_, by intros; rfl⟩
    have equal : partialBuffer (rowStep previous row) previous row.length = rowStep previous row := by
      simpa only [← same] using partialBuffer_finish (rowStep previous row) previous (by simp [same])
    rw [equal]
    exact BlockContinues.range_done (by simp [Calls.rangeContinues]) BlockContinues.nil
termination_by row.length - cursor
decreasing_by omega

def firstElement : Expr := .binOp
  (.subscript (.name "dp") (.binOp (.name "j") .sub (.constant (.int 1)))) .add
  (.subscript (.name "row") (.name "j"))
def firstBody : List Stmt := [.assign [.subscript (.name "dp") (.name "j")] firstElement]

theorem first_element_result (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (address cursor : Nat) (row : List Int)
    (readRow : base[address]? = some (.list (encode row)))
    (positive : 0 < cursor) (inside : cursor < row.length)
    (localRow : env "row" = some (.ref address))
    (localBuffer : env "dp" = some (.ref base.length))
    (localCursor : env "j" = some (number cursor)) :
    Evaluates program env firstElement
      (working base (partialBuffer (firstRow row) (List.replicate row.length 0) cursor))
      (number ((firstRow row).getD cursor 0))
      (working base (partialBuffer (firstRow row) (List.replicate row.length 0) cursor)) := by
  let target := firstRow row
  let original := List.replicate row.length (0 : Int)
  let state := partialBuffer target original cursor
  have size : target.length = row.length := by simp [target, firstRow]
  have before : state[cursor - 1]? = some (target.getD (cursor - 1) 0) := by
    rw [partialBuffer_prefix target original cursor (cursor - 1) (by omega) (by omega)]
    exact getD_some _ _ (by omega)
  have previousIndex : Evaluates program env (.binOp (.name "j") .sub (.constant (.int 1)))
      (working base state) (number (cursor - 1 : Nat)) (working base state) := by
    exact Evaluates.binary (Evaluates.lookup localCursor) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, number, Int.ofNat_sub (by omega : 1 ≤ cursor)])
  have recurrence : target.getD cursor 0 = target.getD (cursor - 1) 0 + row.getD cursor 0 := by
    simpa only [Nat.sub_add_cancel (by omega : 1 ≤ cursor)] using
      firstRowFrom_recurrence 0 row (cursor - 1) (by omega)
  rw [recurrence]
  exact Evaluates.binary
    (index_evaluates program _ env "dp" base.length (cursor - 1) state _ _ localBuffer previousIndex
      (by intros a b c impossible; cases impossible) (found base state) before)
    (index_evaluates program _ env "row" address cursor row _ (.name "j") localRow
      (Evaluates.lookup localCursor) (by intros a b c impossible; cases impossible)
      (prefix_found base state address _ readRow) (getD_some row cursor inside)) rfl

theorem first_update_result (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (address cursor : Nat) (row : List Int)
    (readRow : base[address]? = some (.list (encode row)))
    (positive : 0 < cursor) (inside : cursor < row.length)
    (localRow : env "row" = some (.ref address))
    (localBuffer : env "dp" = some (.ref base.length))
    (localCursor : env "j" = some (number cursor)) :
    BlockContinues program env firstBody
      (working base (partialBuffer (firstRow row) (List.replicate row.length 0) cursor)) env
      (working base (partialBuffer (firstRow row) (List.replicate row.length 0) (cursor + 1))) := by
  have targetLength : (firstRow row).length = (List.replicate row.length (0 : Int)).length := by simp [firstRow]
  apply BlockContinues.setItem (by intros a b c impossible; cases impossible)
    (first_element_result program base env address cursor row readRow positive inside localRow localBuffer localCursor)
    (.cons (Evaluates.lookup localBuffer) (.cons (Evaluates.lookup localCursor) .nil))
  have updated := set_item base (partialBuffer (firstRow row) (List.replicate row.length 0) cursor) cursor
    ((firstRow row).getD cursor 0) (by rw [partialBuffer_length _ _ _ targetLength]; simpa using inside)
  have get : (firstRow row).getD cursor 0 = (firstRow row)[cursor]'(by simp [firstRow]; omega) := by
    simp [List.getD, List.getElem?_eq_getElem (by simp [firstRow]; omega : cursor < (firstRow row).length)]
  rw [get, partialBuffer_step (firstRow row) (List.replicate row.length 0) cursor targetLength (by simp [firstRow]; omega)] at updated
  simpa only [get] using updated

theorem first_loop (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (address cursor : Nat) (row : List Int)
    (readRow : base[address]? = some (.list (encode row)))
    (positive : 0 < cursor) (bounded : cursor ≤ row.length)
    (localRow : env "row" = some (.ref address)) (localBuffer : env "dp" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "j" cursor row.length 1 firstBody []]
      (working base (partialBuffer (firstRow row) (List.replicate row.length 0) cursor)) finalEnv
      (working base (firstRow row)) ∧
      ∀ name, name ≠ "j" → finalEnv name = env name := by
  by_cases active : cursor < row.length
  · let indexed := setLocal env "j" (number cursor)
    have step := first_update_result program base indexed address cursor row readRow positive active
      (by simpa [indexed, setLocal] using localRow) (by simpa [indexed, setLocal] using localBuffer) rfl
    obtain ⟨finalEnv, rest, kept⟩ := first_loop program base indexed address (cursor + 1) row readRow
      (by omega) (by omega) (by simpa [indexed, setLocal] using localRow) (by simpa [indexed, setLocal] using localBuffer)
    refine ⟨finalEnv, ?_, ?_⟩
    · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa [firstBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using step.append rest
    · intro name different
      rw [kept name different]
      simp [indexed, setLocal, different]
  · have done : cursor = row.length := by omega
    subst cursor
    refine ⟨env, ?_, by intros; rfl⟩
    have equal : partialBuffer (firstRow row) (List.replicate row.length 0) row.length = firstRow row := by
      simpa using partialBuffer_finish (firstRow row) (List.replicate row.length 0) (by simp [firstRow])
    rw [equal]
    exact BlockContinues.range_done (by simp [Calls.rangeContinues]) BlockContinues.nil
termination_by row.length - cursor
decreasing_by omega

def firstColumn : List Stmt :=
  [.assign [.subscript (.name "dp") (.constant (.int 0))]
    (.binOp (.subscript (.name "dp") (.constant (.int 0))) .add
      (.subscript (.name "row") (.constant (.int 0))))]

theorem first_column_result (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (address : Nat) (previous row : List Int)
    (readRow : base[address]? = some (.list (encode row)))
    (same : previous.length = row.length) (positive : 0 < row.length)
    (localRow : env "row" = some (.ref address)) (localBuffer : env "dp" = some (.ref base.length)) :
    BlockContinues program env firstColumn (working base previous) env
      (working base (partialBuffer (rowStep previous row) previous 1)) := by
  have prevPositive : 0 < previous.length := by omega
  have size : (rowStep previous row).length = previous.length := by simp [same]
  have first : (rowStep previous row).getD 0 0 = previous.getD 0 0 + row.getD 0 0 := by
    cases previous <;> cases row <;> simp_all [rowStep, List.getD, Int.add_comm]
  apply BlockContinues.setItem (by intros a b c impossible; cases impossible)
    (value := number ((rowStep previous row).getD 0 0))
  · rw [first]
    exact Evaluates.binary
      (index_evaluates program _ env "dp" base.length 0 previous _ (.constant (.int 0)) localBuffer
        Evaluates.constant (by intros a b c impossible; cases impossible) (found base previous) (getD_some previous 0 prevPositive))
      (index_evaluates program _ env "row" address 0 row _ (.constant (.int 0)) localRow
        Evaluates.constant (by intros a b c impossible; cases impossible) (prefix_found base previous address _ readRow)
        (getD_some row 0 positive)) rfl
  · exact .cons (Evaluates.lookup localBuffer) (.cons Evaluates.constant .nil)
  · have updated := set_item base previous 0 ((rowStep previous row).getD 0 0) prevPositive
    have get : (rowStep previous row).getD 0 0 = (rowStep previous row)[0]'(by omega) := by
      simp [List.getD, List.getElem?_eq_getElem (by omega : 0 < (rowStep previous row).length)]
    have step := partialBuffer_step (rowStep previous row) previous 0 size (by omega)
    simp only [partialBuffer_zero, Nat.zero_add] at step
    rw [get, step] at updated
    simpa only [get] using updated

def initialColumn : List Stmt :=
  [.assign [.subscript (.name "dp") (.constant (.int 0))]
    (.subscript (.name "row") (.constant (.int 0)))]

theorem initial_column_result (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (address : Nat) (row : List Int)
    (readRow : base[address]? = some (.list (encode row))) (positive : 0 < row.length)
    (localRow : env "row" = some (.ref address)) (localBuffer : env "dp" = some (.ref base.length)) :
    BlockContinues program env initialColumn (working base (List.replicate row.length 0)) env
      (working base (partialBuffer (firstRow row) (List.replicate row.length 0) 1)) := by
  have size : (firstRow row).length = (List.replicate row.length (0 : Int)).length := by simp [firstRow]
  have first : (firstRow row).getD 0 0 = row.getD 0 0 := by
    cases row <;> simp [firstRow, firstRowFrom, List.getD]
  apply BlockContinues.setItem (by intros a b c impossible; cases impossible)
    (value := number ((firstRow row).getD 0 0))
  · rw [first]
    exact index_evaluates program _ env "row" address 0 row _ (.constant (.int 0)) localRow
      Evaluates.constant (by intros a b c impossible; cases impossible)
      (prefix_found base (List.replicate row.length 0) address _ readRow) (getD_some row 0 positive)
  · exact .cons (Evaluates.lookup localBuffer) (.cons Evaluates.constant .nil)
  · have updated := set_item base (List.replicate row.length 0) 0 ((firstRow row).getD 0 0) (by simpa using positive)
    have get : (firstRow row).getD 0 0 = (firstRow row)[0]'(by simp [firstRow]; omega) := by
      simp [List.getD, List.getElem?_eq_getElem (by simp [firstRow]; omega : 0 < (firstRow row).length)]
    have step := partialBuffer_step (firstRow row) (List.replicate row.length 0) 0 size (by simp [firstRow]; omega)
    simp only [partialBuffer_zero, Nat.zero_add] at step
    rw [get, step] at updated
    simpa only [get] using updated

end ProjectEulerSolutions.GridPathBufferCalls
