import solvers.«11.py.lean.d».ScanRuntime
import lang.py.FormalSW.HeapRuntimeCostSorting

namespace solvers.«11».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime ProjectEulerStatements.Grid

private def directionExpression : Expr := match sourceBody with
  | _ :: _ :: .annAssign _ _ (some expression) :: _ => expression | _ => .constant .none

private theorem directions.allocate (heap : Mutable.Heap) (env : HeapCalls.Env) :
    EvaluatesIn .word program env directionExpression heap 64 (.ref (heap.length + 4)) (heap ++ directionObjects heap) := by
  have first : EvaluatesIn .word program env (.tuple [.constant (.int 0), .constant (.int 1)]) heap 10
      (.ref heap.length) (heap ++ [pairObject .right]) :=
    EvaluatesIn.tuple (.cons EvaluatesIn.constant (.cons EvaluatesIn.constant .nil)) (by simp)
  have second : EvaluatesIn .word program env (.tuple [.constant (.int 1), .constant (.int 0)])
      (heap ++ [pairObject .right]) 10 (.ref (heap.length + 1)) (heap ++ [pairObject .right, pairObject .down]) := by
    simpa [List.append_assoc] using (EvaluatesIn.tuple (module := program) (env := env)
      (heap := heap ++ [pairObject .right])
      (.cons EvaluatesIn.constant (.cons EvaluatesIn.constant .nil))
      (by simp : ([.scalar (.int 1), .scalar (.int 0)] : List Mutable.Value) ≠ []))
  have third : EvaluatesIn .word program env (.tuple [.constant (.int 1), .constant (.int 1)])
      (heap ++ [pairObject .right, pairObject .down]) 10 (.ref (heap.length + 2))
      (heap ++ [pairObject .right, pairObject .down, pairObject .downRight]) := by
    simpa [List.append_assoc] using (EvaluatesIn.tuple (module := program) (env := env)
      (heap := heap ++ [pairObject .right, pairObject .down])
      (.cons EvaluatesIn.constant (.cons EvaluatesIn.constant .nil))
      (by simp : ([.scalar (.int 1), .scalar (.int 1)] : List Mutable.Value) ≠ []))
  have fourth : EvaluatesIn .word program env (.tuple [.constant (.int 1), .unaryOp .uSub (.constant (.int 1))])
      (heap ++ [pairObject .right, pairObject .down, pairObject .downRight]) 13 (.ref (heap.length + 3))
      (heap ++ [pairObject .right, pairObject .down, pairObject .downRight, pairObject .downLeft]) := by
    simpa [List.append_assoc] using (EvaluatesIn.tuple (module := program) (env := env)
      (heap := heap ++ [pairObject .right, pairObject .down, pairObject .downRight])
      (.cons EvaluatesIn.constant (.cons (EvaluatesIn.negate_int (EvaluatesIn.constant (value := .int 1))) .nil))
      (by simp : ([.scalar (.int 1), .scalar (.int (-1))] : List Mutable.Value) ≠ []))
  simpa [directionObjects, directions, List.append_assoc] using
    EvaluatesIn.list (.cons first (.cons second (.cons third (.cons fourth .nil))))

/-- The fixed grid domain bounds every active product; rejected lengths still
pay for all endpoint tests. The input heap is preserved in full. -/
theorem body_run (heap : Heap) (address : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid) (n : Nat)
    (size : grid.length = 20) (rectangle : ∀ row ∈ grid, row.length = 20)
    (initial : HeapCalls.Env) (getGrid : initial "grid" = some (.ref address))
    (getCount : initial "k" = some (.scalar (.int n))) :
    ∃ answer : Int, BlockReturnsIn .word program initial sourceBody heap
      1783566 (.scalar (.int answer)) (heap ++ directionObjects heap) := by
  let withRows := setLocal initial "n" (.scalar (.int grid.length))
  let withColumns := setLocal withRows "m" (.scalar (.int 20))
  let withTable := setLocal withColumns "directions" (.ref (heap.length + 4))
  let withBest := setLocal withTable "best" (.scalar (.int 0))
  have getLength : EvaluatesIn .word program initial (.builtinCall .len [.name "grid"]) heap 5
      (.scalar (.int grid.length)) heap := by
    apply EvaluatesIn.builtin_single (builtin := .len) (work := 1)
      (by intro h; cases h) (by intro h; cases h) (EvaluatesIn.lookup getGrid)
    · simp [HeapCalls.primitive, Mutable.length, view.found, view.size]
    · simp [operation, sequenceLength, view.found]
  have nonempty : 0 < grid.length := by omega
  have firstIndex : 0 < view.rows.length := by rw [view.size]; exact nonempty
  have rowSize : (grid[0]'nonempty).length = 20 := rectangle _ (List.getElem_mem nonempty)
  have readRow : EvaluatesIn .word program withRows
      (.subscript (.name "grid") (.constant (.int 0))) heap 10 (.ref (view.rows[0])) heap := by
    apply EvaluatesIn.list_subscript (by intros a b c h; cases h)
      (EvaluatesIn.lookup (by simpa [withRows, setLocal] using getGrid)) EvaluatesIn.constant view.found
    have read := Mutable.getItem_list heap address 0 (view.rows.map Mutable.Value.ref) view.found (by simpa using firstIndex)
    rw [List.getElem_map] at read
    exact read
  have width : EvaluatesIn .word program withRows
      (.builtinCall .len [.subscript (.name "grid") (.constant (.int 0))]) heap 14
      (.scalar (.int 20)) heap := by
    apply EvaluatesIn.builtin_single (builtin := .len) (work := 1)
      (by intro h; cases h) (by intro h; cases h) readRow
    · simp [HeapCalls.primitive, Mutable.length, view.contents 0 firstIndex, rowSize]
    · simp [operation, sequenceLength, view.contents 0 firstIndex]
  have getWidth : EvaluatesIn .word program withRows
      (.ifExp (.name "n") (.builtinCall .len [.subscript (.name "grid") (.constant (.int 0))]) (.constant (.int 0))) heap 17
      (.scalar (.int 20)) heap := by
    apply EvaluatesIn.conditional (flag := true) (cb := 14) (EvaluatesIn.lookup rfl)
    · simp [Mutable.truth, scalarTruth, nonempty.ne']
    · exact width
  obtain ⟨finalEnv, answer, scan, result, _⟩ := rows.loop (heap ++ directionObjects heap) address (heap.length + 4)
    grid (view.append _) (directionView heap) 20 n 0 size rfl rectangle withBest 0
    (by simpa [withBest, withTable, withColumns, withRows, setLocal] using getGrid)
    (by simpa [withBest, withTable, withColumns, withRows, setLocal] using getCount)
    rfl rfl rfl rfl
  have loop : BlockContinuesIn .word program withBest
      [.forIn (.name "i") (.call (.name "range") [.name "n"]) rowsBody []]
      (heap ++ directionObjects heap) 1783469 finalEnv (heap ++ directionObjects heap) := by
    have run := BlockContinuesIn.for_range_bound (EvaluatesIn.lookup (name := "n") (env := withBest) rfl) scan
    exact run.mono (by rw [size])
  have returned : BlockReturnsIn .word program finalEnv [.return (.name "best")]
      (heap ++ directionObjects heap) 2 (.scalar (.int answer)) (heap ++ directionObjects heap) :=
    BlockReturnsIn.return_value (EvaluatesIn.lookup result)
  have bodyRun : BlockReturnsIn .word program initial sourceBody heap 1783566
      (.scalar (.int answer)) (heap ++ directionObjects heap) := by
    apply BlockReturnsIn.assign (cb := 1783559) getLength
    apply BlockReturnsIn.assign (cb := 1783540) getWidth
    apply BlockContinuesIn.returns (cb := 1783474) (BlockContinuesIn.annAssign (directions.allocate heap withColumns))
    apply BlockReturnsIn.assign (cb := 1783471) EvaluatesIn.constant
    exact loop.returns returned
  exact ⟨answer, bodyRun⟩

end solvers.«11».runtimeCost
