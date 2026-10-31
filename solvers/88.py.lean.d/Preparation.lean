import solvers.«88.py.lean.d».Allocation

namespace solvers.«88».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

theorem small_result (maxSize : Int) (small : maxSize < 2) (heap : Mutable.Heap) :
    ReturnsValue program "solve" [integer maxSize] heap (number 0)
      (heap ++ [.list [integer maxSize]]) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := .ref heap.length) (middle := heap ++ [.list [integer maxSize]])
  · exact Evaluates.list (.cons (Evaluates.lookup rfl) .nil)
  · apply BlockReturns.branch (flag := true) (value := .scalar (.bool true))
      (middle := heap ++ [.list [integer maxSize]])
    · apply Evaluates.compare (av := integer maxSize) (bv := number 2)
      · apply Evaluates.subscript (by intros; simp) (Evaluates.lookup rfl) Evaluates.constant
        exact Mutable.getItem_list (heap ++ [.list [integer maxSize]]) heap.length 0 [integer maxSize]
          (by simp) (by simp)
      · exact Evaluates.constant
      · simp [HeapCalls.compare, scalarCompare, evalCmpOp, small]
    · rfl
    · exact BlockReturns.return_value Evaluates.constant

/-- The real closure setup and complete recursive call produce the proved minima table. -/
theorem prepare (maxSize : Nat) (large : 2 ≤ maxSize) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (getMax : env "k_max" = some (number maxSize)) :
    BlockContinues program env (code.take 6) heap (preparedEnv env heap maxSize)
      (working (context heap maxSize) (table maxSize)) := by
  let maxEnv := setLocal env "__formalsw_cell_solve_k_max" (.ref heap.length)
  let limitEnv := setLocal maxEnv "__formalsw_cell_solve_limit" (.ref (heap.length + 1))
  let sentinelEnv := setLocal limitEnv "sentinel" (number (2 * maxSize + 1))
  let ready := preparedEnv env heap maxSize
  let h1 := heap ++ [.list [number maxSize]]
  let h2 := heap ++ [.list [number maxSize], .list [number (2 * maxSize)]]
  have first : BlockContinues program env [code[0]'(by decide)] heap maxEnv h1 :=
    BlockContinues.assign (Evaluates.list (.cons (Evaluates.lookup getMax) .nil))
  have guard : BlockContinues program maxEnv [code[1]'(by decide)] h1 maxEnv h1 := by
    apply BlockContinues.branch (flag := false) (value := .scalar (.bool false)) (middle := h1)
    · apply Evaluates.compare (av := number maxSize) (bv := number 2)
        (captured_number h1 heap.length maxSize "__formalsw_cell_solve_k_max" maxEnv
          (by simp [maxEnv, setLocal]) (by simp [h1])) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, show ¬ (maxSize : Int) < 2 by omega]
    · rfl
    · exact .nil
  have bound : BlockContinues program maxEnv [code[2]'(by decide)] h1 limitEnv h2 := by
    apply BlockContinues.assign
    have allocated := Evaluates.list (Evaluates.Arguments.cons
      (Evaluates.binary (module := program) (env := maxEnv) (heap := h1) (av := number 2) (bv := number maxSize)
        Evaluates.constant
        (captured_number h1 heap.length maxSize "__formalsw_cell_solve_k_max" maxEnv
          (by simp [maxEnv, setLocal]) (by simp [h1]))
        (show Mutable.binary .mult (number 2) (number maxSize) h1 = some (.ok (number (2 * maxSize)), h1) by
          simp [Mutable.binary, scalarBinary, number])) .nil)
    simpa [h1, h2, List.append_assoc] using allocated
  have sentinel : BlockContinues program limitEnv [code[3]'(by decide)] h2 sentinelEnv h2 := by
    apply BlockContinues.assign
    exact Evaluates.binary (av := number (2 * maxSize)) (bv := number 1)
      (captured_number h2 (heap.length + 1) (2 * maxSize) "__formalsw_cell_solve_limit" limitEnv
        (by simp [limitEnv, setLocal]) (by simp [h2])) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, number])
  have best : BlockContinues program sentinelEnv [code[4]'(by decide)] h2 ready
      (working (context heap maxSize) (List.replicate (maxSize + 1) (2 * maxSize + 1))) := by
    apply BlockContinues.assign
    have allocated := allocate_best maxSize h2 heap.length sentinelEnv
      (by simp [sentinelEnv, limitEnv, maxEnv, setLocal]) (by simp [h2]) (by simp [sentinelEnv, setLocal])
    simpa [h2, context, working, ProjectEulerSolutions.HeapCollection.working, List.append_assoc,
      ready, preparedEnv, sentinelEnv, limitEnv, maxEnv] using allocated
  have searched : BlockContinues program ready [code[5]'(by decide)]
      (working (context heap maxSize) (List.replicate (maxSize + 1) (2 * maxSize + 1))) ready
      (working (context heap maxSize) (table maxSize)) := by
    apply BlockContinues.discard
    apply Evaluates.call (values := [number 2, number 1, number 0, number 0,
      .ref (context heap maxSize).bestCell, .ref (context heap maxSize).maxCell, .ref (context heap maxSize).limitCell])
    · exact .cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant
        (.cons (Evaluates.lookup (by simp [ready, preparedEnv, context, setLocal]))
        (.cons (Evaluates.lookup (by simp [ready, preparedEnv, context, setLocal]))
        (.cons (Evaluates.lookup (by simp [ready, preparedEnv, context, setLocal])) .nil))))))
    · exact dfs.root maxSize (context heap maxSize) (context_cells heap maxSize)
  exact first.append (guard.append (bound.append (sentinel.append (best.append searched))))

end solvers.«88».solve
