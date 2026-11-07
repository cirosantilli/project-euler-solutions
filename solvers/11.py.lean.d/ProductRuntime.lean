import solvers.«11.py.lean.d».Source
import lang.py.FormalSW.HeapRuntimeCostSetup
import lang.py.FormalSW.HeapRuntimeCostComparisons
import lang.py.FormalSW.HeapRuntimeCostBooleans

namespace solvers.«11».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime ProjectEulerStatements.Grid

/-- Two direct indexed reads charge both bounds/index calculations. -/
theorem grid_read_cost (heap : Heap) (address : Nat) (grid : List (List Nat))
    (view : GridView heap address grid) (env : HeapCalls.Env) (r c : Int)
    (rowExpr columnExpr : Expr) (cr cc : Nat)
    (ordinaryRow : ∀ a b c, rowExpr ≠ .slice a b c)
    (ordinaryColumn : ∀ a b c, columnExpr ≠ .slice a b c)
    (getGrid : env "grid" = some (.ref address))
    (getRow : EvaluatesIn .word program env rowExpr heap cr (.scalar (.int r)) heap)
    (getColumn : EvaluatesIn .word program env columnExpr heap cc (.scalar (.int c)) heap)
    (rowValid : 0 ≤ r ∧ r < grid.length)
    (columnValid : 0 ≤ c ∧ c < (grid[r.toNat]'(by omega)).length) :
    ∃ value : Int, EvaluatesIn .word program env (.subscript (.subscript (.name "grid") rowExpr) columnExpr)
      heap (cr+cc+17) (.scalar (.int value)) heap := by
  have rowIndex : r.toNat < grid.length := by omega
  have addressIndex : r.toNat < view.rows.length := by rw [view.size]; exact rowIndex
  let row := view.rows[r.toNat]
  let values := grid[r.toNat]
  have contents := view.contents r.toNat addressIndex
  have readRow : EvaluatesIn .word program env (.subscript (.name "grid") rowExpr) heap (cr+9) (.ref row) heap := by
    apply EvaluatesIn.mono (small := 1+cr+1+1+6) (larger := by omega)
    apply EvaluatesIn.list_subscript ordinaryRow (EvaluatesIn.lookup getGrid) getRow view.found
    have read := Mutable.getItem_list heap address r.toNat (view.rows.map Mutable.Value.ref) view.found (by simpa using addressIndex)
    rw [List.getElem_map] at read
    simpa [Int.toNat_of_nonneg rowValid.1, row] using read
  change 0 ≤ c ∧ c < values.length at columnValid
  have columnIndex : c.toNat < values.length := by omega
  refine ⟨values[c.toNat], ?_⟩
  apply EvaluatesIn.mono (small := (cr+9)+cc+1+1+6) (larger := by omega)
  apply EvaluatesIn.list_subscript ordinaryColumn readRow getColumn contents
  have read := Mutable.getItem_list heap row c.toNat
    (values.map (fun (n : Nat) => .scalar (.int (n : Int)))) contents (by simpa using columnIndex)
  rw [List.getElem_map] at read
  simpa [Int.toNat_of_nonneg columnValid.1] using read

private def productRow : Expr := match productExpr with | .subscript (.subscript _ row) _ => row | _ => .constant .none
private def productColumn : Expr := match productExpr with | .subscript _ column => column | _ => .constant .none

private theorem product_loop (direction : Direction) (heap : Heap) (address : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid) (columns r c n : Nat)
    (rectangle : ∀ row ∈ grid, row.length = columns) (valid : ValidStart direction grid.length columns n r c)
    (remaining position : Nat) (endpoint : remaining+position = n) (env : HeapCalls.Env) (acc : Int)
    (getGrid : env "grid" = some (.ref address)) (getRow : env "i" = some (.scalar (.int r)))
    (getColumn : env "j" = some (.scalar (.int c)))
    (getDi : env "di" = some (.scalar (.int direction.rowStep)))
    (getDj : env "dj" = some (.scalar (.int direction.columnStep)))
    (getAcc : env "prod" = some (.scalar (.int acc))) :
    ∃ finalEnv, ∃ answer : Int,
      BlockContinuesIn .word program env [.rangeStepLoop "t" position n 1 productBody []] heap
        (49*remaining+4) finalEnv heap ∧ finalEnv "prod" = some (.scalar (.int answer)) ∧
      (∀ name, name ≠ "t" → name ≠ "prod" → finalEnv name = env name) := by
  induction remaining generalizing position env acc with
  | zero =>
      refine ⟨env, acc, ?_, getAcc, by intros; rfl⟩
      exact BlockContinuesIn.range_done (by simp [Calls.rangeContinues]; omega) BlockContinuesIn.nil
  | succ remaining ih =>
      let indexed := setLocal env "t" (.scalar (.int position))
      have savedGrid : indexed "grid" = some (.ref address) := by simpa [indexed, setLocal] using getGrid
      have savedRow : indexed "i" = some (.scalar (.int r)) := by simpa [indexed, setLocal] using getRow
      have savedColumn : indexed "j" = some (.scalar (.int c)) := by simpa [indexed, setLocal] using getColumn
      have savedDi : indexed "di" = some (.scalar (.int direction.rowStep)) := by simpa [indexed, setLocal] using getDi
      have savedDj : indexed "dj" = some (.scalar (.int direction.columnStep)) := by simpa [indexed, setLocal] using getDj
      have savedAcc : indexed "prod" = some (.scalar (.int acc)) := by simpa [indexed, setLocal] using getAcc
      have row : EvaluatesIn .word program indexed productRow heap 11 (.scalar (.int (direction.row r position))) heap := by
        rw [Direction.row_eq]
        exact EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup savedRow)
          (EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup rfl) (EvaluatesIn.lookup savedDi) rfl rfl) rfl rfl
      have column : EvaluatesIn .word program indexed productColumn heap 11 (.scalar (.int (direction.column c position))) heap := by
        rw [Direction.column_eq]
        exact EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup savedColumn)
          (EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup rfl) (EvaluatesIn.lookup savedDj) rfl rfl) rfl rfl
      have rowBounds : 0 ≤ direction.row r position ∧ direction.row r position < grid.length := by
        rw [Direction.row_eq]
        exact segment_bounds r direction.rowStep grid.length n position (by have := valid.1; constructor <;> omega)
          valid.2.2.1 (by omega)
      have columnBounds : 0 ≤ direction.column c position ∧ direction.column c position < columns := by
        rw [Direction.column_eq]
        exact segment_bounds c direction.columnStep columns n position (by have := valid.2.1; constructor <;> omega)
          valid.2.2.2 (by omega)
      have rowSize : (grid[(direction.row r position).toNat]'(by omega)).length = columns := rectangle _ (List.getElem_mem _)
      obtain ⟨value, read⟩ := grid_read_cost heap address grid view indexed _ _ productRow productColumn 11 11
        (by intros a b c h; cases h) (by intros a b c h; cases h) savedGrid row column rowBounds
        (by simpa only [rowSize] using columnBounds)
      let nextEnv := setLocal indexed "prod" (.scalar (.int (acc*value)))
      have body : BlockContinuesIn .word program indexed productBody heap 45 nextEnv heap :=
        BlockContinuesIn.augAssign (op := .mult) (work := 1) savedAcc read rfl rfl
      obtain ⟨finalEnv, answer, rest, finalAcc, preserved⟩ := ih (position+1) (by omega) nextEnv (acc*value)
        (by simpa [nextEnv, setLocal] using savedGrid) (by simpa [nextEnv, setLocal] using savedRow)
        (by simpa [nextEnv, setLocal] using savedColumn) (by simpa [nextEnv, setLocal] using savedDi)
        (by simpa [nextEnv, setLocal] using savedDj) rfl
      refine ⟨finalEnv, answer, ?_, finalAcc, ?_⟩
      · have run := BlockContinuesIn.range_step (env := env) (name := "t")
          (by simp [Calls.rangeContinues]; omega : Calls.rangeContinues (position : Int) n 1 = true)
          (by simpa only [productBody, candidateBody, directionBody, columnsBody, rowsBody, sourceBody,
            Calls.loopContinuation, Nat.cast_add, Nat.cast_one] using body.append rest)
        exact run.mono (by omega)
      · intro name notT notProduct
        rw [preserved name notT notProduct]
        simp [nextEnv, indexed, setLocal, notT, notProduct]

private def updateBest : List Stmt := match candidateBody with | [_, _, statement] => [statement] | _ => []

private theorem update_best (heap : Heap) (env : HeapCalls.Env) (best value : Int)
    (getBest : env "best" = some (.scalar (.int best))) (getValue : env "prod" = some (.scalar (.int value))) :
    ∃ finalEnv, ∃ answer : Int, BlockContinuesIn .word program env updateBest heap 12 finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int answer)) ∧ (∀ name, name ≠ "best" → finalEnv name = env name) := by
  have compared : EvaluatesIn .word program env (.compare (.name "prod") [(.gt, .name "best")]) heap 7
      (.scalar (.bool (decide (best < value)))) heap := by
    apply EvaluatesIn.compare (work := 2) (EvaluatesIn.lookup getValue) (EvaluatesIn.lookup getBest)
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · rfl
  by_cases larger : best < value
  · refine ⟨setLocal env "best" (.scalar (.int value)), value, ?_, by simp [setLocal], ?_⟩
    · exact BlockContinuesIn.branch (flag := true) (by simpa [larger] using compared) rfl
        (BlockContinuesIn.assign (EvaluatesIn.lookup getValue))
    · intro name other; simp [setLocal, other]
  · refine ⟨env, best, ?_, getBest, by intros; rfl⟩
    exact (BlockContinuesIn.branch (flag := false) (by simpa [larger] using compared) rfl BlockContinuesIn.nil).mono (by omega)

theorem candidate_cost (direction : Direction) (heap : Heap) (address : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid) (columns r c n : Nat)
    (rectangle : ∀ row ∈ grid, row.length = columns) (valid : ValidStart direction grid.length columns n r c)
    (env : HeapCalls.Env) (best : Int)
    (getGrid : env "grid" = some (.ref address)) (getRow : env "i" = some (.scalar (.int r)))
    (getColumn : env "j" = some (.scalar (.int c)))
    (getDi : env "di" = some (.scalar (.int direction.rowStep))) (getDj : env "dj" = some (.scalar (.int direction.columnStep)))
    (getCount : env "k" = some (.scalar (.int n))) (getBest : env "best" = some (.scalar (.int best))) :
    ∃ finalEnv, ∃ answer : Int, BlockContinuesIn .word program env candidateBody heap (49*n+24) finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int answer)) ∧
      (∀ name, name ≠ "t" → name ≠ "prod" → name ≠ "best" → finalEnv name = env name) := by
  let initial := setLocal env "prod" (.scalar (.int 1))
  obtain ⟨middle, value, loop, getValue, preserved⟩ := product_loop direction heap address grid view columns r c n rectangle valid
    n 0 (by omega) initial 1
    (by simpa [initial, setLocal] using getGrid) (by simpa [initial, setLocal] using getRow)
    (by simpa [initial, setLocal] using getColumn) (by simpa [initial, setLocal] using getDi)
    (by simpa [initial, setLocal] using getDj) rfl
  have entered := BlockContinuesIn.for_range_bound (EvaluatesIn.lookup (name := "k")
    (by simpa [initial, setLocal] using getCount)) loop
  obtain ⟨finalEnv, answer, updated, result, saved⟩ := update_best heap middle best value
    (by rw [preserved "best" (by decide) (by decide)]; simpa [initial, setLocal] using getBest) getValue
  refine ⟨finalEnv, answer, ?_, result, ?_⟩
  · exact ((BlockContinuesIn.assign (name := "prod") EvaluatesIn.constant).append (entered.append updated)).mono (by omega)
  · intro name notT notProd notBest
    rw [saved name notBest, preserved name notT notProd]
    simp [initial, setLocal, notProd]

end solvers.«11».runtimeCost
