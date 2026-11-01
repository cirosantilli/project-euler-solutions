import ProjectEulerSolutions.P82
import ProjectEulerStatements.Lib.GridPathColumns

/-! In-place vertical sweeps implement the list-based column recurrences. -/
namespace ProjectEulerSolutions.P82
open ProjectEulerStatements.P81 (partialBuffer)
open ProjectEulerStatements.P82
set_option Elab.async false

theorem down_update (cost column : List Int) (index : Nat) (same : cost.length = column.length)
    (positive : 0 < index) (inside : index < cost.length) :
    downUpdate column (partialBuffer (relaxDown cost column) cost index) index =
      partialBuffer (relaxDown cost column) cost (index + 1) := by
  have size := relaxDown_length cost column same
  have recurrence : (relaxDown cost column).getD index 0 =
      min (cost.getD index 0) ((relaxDown cost column).getD (index - 1) 0 + column.getD index 0) := by
    simpa only [Nat.sub_add_cancel (by omega : 1 ≤ index)] using
      relaxDown_recurrence cost column (index - 1) (by omega) same
  unfold downUpdate
  rw [ProjectEulerStatements.P81.partialBuffer_getD_suffix _ _ _ _ (by omega) (by omega),
    ProjectEulerStatements.P81.partialBuffer_getD_prefix _ _ _ _ (by omega) (by omega), ← recurrence]
  have get : (relaxDown cost column).getD index 0 = (relaxDown cost column)[index]'(by omega) := by
    simp [List.getD, List.getElem?_eq_getElem (by omega : index < (relaxDown cost column).length)]
  rw [get, ProjectEulerStatements.P81.partialBuffer_step _ _ index size (by omega)]

theorem downBuffer_eq (cost column : List Int) (same : cost.length = column.length) :
    downBuffer cost column = relaxDown cost column := by
  cases cost with
  | nil =>
      have empty : column = [] := by simpa using same.symm
      simp [downBuffer, relaxDown, empty]
  | cons c cs =>
      cases column with
      | nil => simp at same
      | cons v vs =>
          have start : partialBuffer (relaxDown (c :: cs) (v :: vs)) (c :: cs) 1 = c :: cs := by
            simp [partialBuffer, relaxDown, same]
          unfold downBuffer
          rw [if_pos same]
          nth_rw 1 [← start]
          exact ProjectEulerStatements.P81.fold_buffer _ _ _ 1 cs.length (by simp; omega)
            (relaxDown_length _ _ same) (fun index after valid => down_update _ _ index same (by omega) valid)

theorem reverse_set (values : List Int) (index : Nat) (value : Int) (inside : index < values.length) :
    (values.set index value).reverse = values.reverse.set (values.length - 1 - index) value := by
  apply List.ext_getElem (by simp)
  intro j left right
  simp only [List.getElem_reverse, List.getElem_set, List.length_set]
  have bounded : j < values.length := by simpa using left
  split_ifs <;> first | rfl | omega

theorem up_update_reverse (column buffer : List Int) (index : Nat)
    (same : buffer.length = column.length) (positive : 0 < index) (inside : index < buffer.length) :
    (upUpdate column buffer (buffer.length - 1 - index)).reverse =
      downUpdate column.reverse buffer.reverse index := by
  have mirror : buffer.length - 1 - index < buffer.length := by omega
  have reflected : buffer.length - 1 - (buffer.length - 1 - index) = index := by omega
  have adjacent : buffer.length - 1 - (index - 1) = buffer.length - 1 - index + 1 := by omega
  unfold upUpdate downUpdate
  rw [reverse_set buffer _ _ mirror, reflected, List.getD_reverse index inside,
    List.getD_reverse (index - 1) (by omega : index - 1 < buffer.length),
    List.getD_reverse index (by omega : index < column.length)]
  rw [← same, adjacent]

theorem up_fold_reverse (indices : List Nat) (column buffer : List Int) (size : Nat)
    (same : buffer.length = column.length) (length : buffer.length = size)
    (bounds : ∀ index ∈ indices, 0 < index ∧ index < size) :
    (indices.foldl (fun values index => upUpdate column values (size - 1 - index)) buffer).reverse =
      indices.foldl (downUpdate column.reverse) buffer.reverse := by
  induction indices generalizing buffer with
  | nil => rfl
  | cons index indices ih =>
      have bound := bounds index (by simp)
      simp only [List.foldl_cons]
      rw [ih _ (by simpa [upUpdate] using same) (by simpa [upUpdate] using length)
        (fun i member => bounds i (by simp [member]))]
      rw [← length, up_update_reverse column buffer index same bound.1 (by omega)]

theorem upBuffer_eq (cost column : List Int) (same : cost.length = column.length) :
    upBuffer cost column = relaxUp cost column := by
  have mirrored := up_fold_reverse (List.range' 1 (cost.length - 1)) column cost cost.length same rfl
    (by intro index member; have bound := List.mem_range'.mp member; omega)
  have down := downBuffer_eq cost.reverse column.reverse (by simpa using same)
  simp only [downBuffer, List.length_reverse, if_pos same] at down
  unfold upBuffer relaxUp
  rw [if_pos same]
  apply List.reverse_inj.mp
  simpa only [List.reverse_reverse] using mirrored.trans down

theorem upBuffer_reverse_range (cost column : List Int) (same : cost.length = column.length) :
    upBuffer cost column = (List.range (cost.length - 1)).reverse.foldl (upUpdate column) cost := by
  unfold upBuffer
  rw [if_pos same, List.range'_eq_map_range, List.foldl_map]
  conv_rhs => rw [List.range_eq_range', List.reverse_range', List.foldl_map]
  congr 1
  funext buffer index
  congr 1
  omega

end ProjectEulerSolutions.P82
