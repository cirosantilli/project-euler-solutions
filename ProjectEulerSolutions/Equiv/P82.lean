import ProjectEulerSolutions.Lib.GridPathColumnBuffers

namespace ProjectEulerSolutions.P82
open ProjectEulerStatements.P82
set_option Elab.async false

theorem nextColumn_eq (previous column : List Int) (same : previous.length = column.length) :
    nextColumn previous column = columnStep previous column := by
  have added := addLists_length previous column same
  have down := relaxDown_length (addLists previous column) column (by omega)
  unfold nextColumn columnStep
  rw [show List.zipWith (· + ·) previous column = addLists previous column by simp [addLists, same]]
  dsimp only
  rw [downBuffer_eq _ column (by omega), upBuffer_eq _ column (by omega)]

theorem columns_fold (indices : List Nat) (matrix : List (List Int)) (buffer : List Int)
    (size : buffer.length = matrix.length) :
    indices.foldl (fun values column => nextColumn values (getCol matrix column)) buffer =
      (indices.map (getCol matrix)).foldl columnStep buffer := by
  induction indices generalizing buffer with
  | nil => rfl
  | cons index indices ih =>
      simp only [List.foldl_cons, List.map_cons]
      rw [nextColumn_eq buffer _ (by simpa using size)]
      apply ih
      exact (columnStep_length buffer _ (by simpa using size)).trans size

theorem equiv (matrix : List (List Int)) : ProjectEulerStatements.P82.naive matrix = solve matrix := by
  unfold ProjectEulerStatements.P82.naive minPath solve minimalPathSumThreeWays
  cases width : (matrix.headD []).length with
  | zero => simp
  | succ n =>
      simp only [Nat.add_one_ne_zero, if_false, Nat.add_sub_cancel, List.range_eq_range', List.range'_succ,
        List.map_cons]
      rw [columns_fold _ matrix _ (getCol_length matrix 0)]

end ProjectEulerSolutions.P82
