import ProjectEulerSolutions.P81
import ProjectEulerStatements.Lib.GridPathRows

namespace ProjectEulerSolutions.P81
open ProjectEulerStatements.P81
set_option Elab.async false

theorem first_update (row : List Int) (index : Nat) (positive : 0 < index) (inside : index < row.length) :
    firstUpdate row (partialBuffer (firstRow row) (List.replicate row.length 0) index) index =
      partialBuffer (firstRow row) (List.replicate row.length 0) (index + 1) := by
  have size : (firstRow row).length = row.length := by simp [firstRow]
  have recurrence : (firstRow row).getD index 0 =
      (firstRow row).getD (index - 1) 0 + row.getD index 0 := by
    simpa only [Nat.sub_add_cancel (by omega : 1 ≤ index)] using
      firstRowFrom_recurrence 0 row (index - 1) (by omega)
  unfold firstUpdate
  rw [partialBuffer_getD_prefix _ _ _ _ (by omega) (by omega), ← recurrence]
  have get : (firstRow row).getD index 0 = (firstRow row)[index]'(by omega) := by
    simp [List.getD, List.getElem?_eq_getElem (by omega : index < (firstRow row).length)]
  rw [get, partialBuffer_step (firstRow row) (List.replicate row.length 0) index (by simpa using size) (by omega)]

theorem row_update (previous row : List Int) (index : Nat) (same : previous.length = row.length)
    (positive : 0 < index) (inside : index < row.length) :
    rowUpdate row (partialBuffer (rowStep previous row) previous index) index =
      partialBuffer (rowStep previous row) previous (index + 1) := by
  have size : (rowStep previous row).length = previous.length := by simp [same]
  have recurrence : (rowStep previous row).getD index 0 = row.getD index 0 +
      min (previous.getD index 0) ((rowStep previous row).getD (index - 1) 0) := by
    simpa only [Nat.sub_add_cancel (by omega : 1 ≤ index)] using
      rowStep_recurrence previous row (index - 1) (by omega) same
  unfold rowUpdate
  rw [partialBuffer_getD_suffix _ _ _ _ (by omega) (by omega),
    partialBuffer_getD_prefix _ _ _ _ (by omega) (by omega), ← recurrence]
  have get : (rowStep previous row).getD index 0 = (rowStep previous row)[index]'(by omega) := by
    simp [List.getD, List.getElem?_eq_getElem (by omega : index < (rowStep previous row).length)]
  rw [get, partialBuffer_step (rowStep previous row) previous index size (by omega)]

theorem firstBuffer_eq (row : List Int) : firstBuffer row = firstRow row := by
  cases row with
  | nil => rfl
  | cons x xs =>
      have start : (List.replicate (x :: xs).length (0 : Int)).set 0 x =
          partialBuffer (firstRow (x :: xs)) (List.replicate (x :: xs).length 0) 1 := by
        simp [partialBuffer, firstRow, firstRowFrom, List.replicate_succ]
      simp only [firstBuffer]
      rw [start]
      exact fold_buffer _ _ _ 1 xs.length (by simp [Nat.add_comm])
        (by simp [firstRow]) (fun index after valid => first_update _ index (by omega) (by simpa using valid))

theorem nextBuffer_eq (previous row : List Int) (same : previous.length = row.length) :
    nextBuffer previous row = rowStep previous row := by
  cases previous with
  | nil => cases row <;> rfl
  | cons p ps =>
      cases row with
      | nil => rfl
      | cons x xs =>
          have start : (p :: ps).set 0 (p + x) = partialBuffer (rowStep (p :: ps) (x :: xs)) (p :: ps) 1 := by
            simp [partialBuffer, rowStep, Int.add_comm]
          simp only [nextBuffer]
          rw [start]
          exact fold_buffer _ _ _ 1 xs.length (by simp only [List.length_cons] at same ⊢; omega)
            (by simp [same]) (fun index after valid => row_update _ _ index same (by omega) (by omega))

theorem fold_rows (rows : List (List Int)) (buffer : List Int) (width : Nat)
    (shape : ∀ row ∈ rows, row.length = width) (size : buffer.length = width) :
    rows.foldl nextBuffer buffer = rows.foldl rowStep buffer := by
  induction rows generalizing buffer with
  | nil => rfl
  | cons row rows ih =>
      simp only [List.foldl_cons]
      rw [nextBuffer_eq buffer row (by rw [size, shape row (by simp)])]
      exact ih _ (fun r member => shape r (by simp [member])) (by simp [size, shape row (by simp)])

theorem equiv (matrix : List (List Int))
    (rectangular : ∀ row ∈ matrix, row.length = (matrix.headD []).length) :
    ProjectEulerStatements.P81.naive matrix = solve matrix := by
  cases matrix with
  | nil => rfl
  | cons row rows =>
      simp only [solve, minPathSum, ProjectEulerStatements.P81.naive, minPath]
      rw [firstBuffer_eq, fold_rows rows (firstRow row) row.length
        (fun r member => rectangular r (by simp [member])) (by simp [firstRow])]

end ProjectEulerSolutions.P81
