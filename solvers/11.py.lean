import «.formalsw».solvers.«11.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p11.py»
import ProjectEulerStatements.P11
import FormalSW

namespace solvers.«11»
open FormalSW.PyAst
open ProjectEulerStatements.Grid
private abbrev program := Solvers.P11Ast.module
abbrev GridView := data.«project-euler-statements».data.py.project_euler_statements.p11.GridView

/-- Direct Python indexing agrees with the padded mathematical accessor whenever
both coordinates are inside the represented grid. -/
theorem grid_read (heap : Mutable.Heap) (address : Nat) (grid : List (List Nat))
    (view : GridView heap address grid) (env : HeapCalls.Env) (r c : Int)
    (rowExpr columnExpr : Expr)
    (ordinaryRow : ∀ lower upper stride, rowExpr ≠ .slice lower upper stride)
    (ordinaryColumn : ∀ lower upper stride, columnExpr ≠ .slice lower upper stride)
    (getGrid : env "grid" = some (.ref address))
    (getRow : HeapCalls.Evaluates program env rowExpr heap (.scalar (.int r)) heap)
    (getColumn : HeapCalls.Evaluates program env columnExpr heap (.scalar (.int c)) heap)
    (rowValid : 0 ≤ r ∧ r < grid.length)
    (columnValid : 0 ≤ c ∧ c < (grid[r.toNat]'(by omega)).length) :
    HeapCalls.Evaluates program env (.subscript (.subscript (.name "grid") rowExpr) columnExpr)
      heap (.scalar (.int (↑(ProjectEulerStatements.Grid.get grid r c 0)))) heap := by
  have rowIndex : r.toNat < grid.length := by omega
  have addressIndex : r.toNat < view.rows.length := by rw [view.size]; exact rowIndex
  let row := view.rows[r.toNat]
  let values := grid[r.toNat]
  have contents := view.contents r.toNat addressIndex
  have readRow : HeapCalls.Evaluates program env (.subscript (.name "grid") rowExpr) heap (.ref row) heap := by
    apply HeapCalls.Evaluates.subscript ordinaryRow
      (HeapCalls.Evaluates.lookup getGrid) getRow
    have read := Mutable.getItem_list heap address r.toNat (view.rows.map Mutable.Value.ref)
      view.found (by simpa using addressIndex)
    rw [List.getElem_map] at read
    simpa [Int.toNat_of_nonneg rowValid.1, row] using read
  change 0 ≤ c ∧ c < values.length at columnValid
  have columnIndex : c.toNat < values.length := by omega
  have result : ProjectEulerStatements.Grid.get grid r c 0 = values[c.toNat] := by
    simp [ProjectEulerStatements.Grid.get, rowValid.1, columnValid.1,
      List.getElem?_eq_getElem rowIndex, values, List.getElem?_eq_getElem columnIndex]
  rw [result]
  apply HeapCalls.Evaluates.subscript ordinaryColumn readRow getColumn
  have read := Mutable.getItem_list heap row c.toNat
    (values.map (fun (n : Nat) => .scalar (.int (n : Int)))) contents (by simpa using columnIndex)
  rw [List.getElem_map] at read
  simpa [Int.toNat_of_nonneg columnValid.1] using read

private abbrev productRow : Expr :=
  .binOp (.name "i") .add (.binOp (.name "t") .mult (.name "di"))
private abbrev productColumn : Expr :=
  .binOp (.name "j") .add (.binOp (.name "t") .mult (.name "dj"))
private abbrev productBody : List Stmt :=
  [.augAssign (.name "prod") .mult (.subscript (.subscript (.name "grid") productRow) productColumn)]

private theorem product.loop (direction : Direction) (heap : Mutable.Heap) (address : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid) (columns r c n position : Nat)
    (rectangle : ∀ row ∈ grid, row.length = columns)
    (valid : ValidStart direction grid.length columns n r c) (env : HeapCalls.Env) (acc : Int)
    (getGrid : env "grid" = some (.ref address))
    (getRow : env "i" = some (.scalar (.int r)))
    (getColumn : env "j" = some (.scalar (.int c)))
    (getDi : env "di" = some (.scalar (.int direction.rowStep)))
    (getDj : env "dj" = some (.scalar (.int direction.columnStep)))
    (getAcc : env "prod" = some (.scalar (.int acc))) :
    ∃ finalEnv, HeapCalls.BlockContinues program env
      [.rangeStepLoop "t" position n 1 productBody []] heap finalEnv heap ∧
      finalEnv "prod" = some (.scalar (.int (acc * segmentTail grid direction r c position n))) ∧
      (∀ name, name ≠ "t" → name ≠ "prod" → finalEnv name = env name) := by
  by_cases active : position < n
  · let indexed := HeapCalls.setLocal env "t" (.scalar (.int position))
    have savedGrid : indexed "grid" = some (.ref address) := by simpa [indexed, HeapCalls.setLocal] using getGrid
    have savedRow : indexed "i" = some (.scalar (.int r)) := by simpa [indexed, HeapCalls.setLocal] using getRow
    have savedColumn : indexed "j" = some (.scalar (.int c)) := by simpa [indexed, HeapCalls.setLocal] using getColumn
    have savedDi : indexed "di" = some (.scalar (.int direction.rowStep)) := by simpa [indexed, HeapCalls.setLocal] using getDi
    have savedDj : indexed "dj" = some (.scalar (.int direction.columnStep)) := by simpa [indexed, HeapCalls.setLocal] using getDj
    have savedAcc : indexed "prod" = some (.scalar (.int acc)) := by simpa [indexed, HeapCalls.setLocal] using getAcc
    have row : HeapCalls.Evaluates program indexed productRow heap (.scalar (.int (direction.row r position))) heap := by
      rw [Direction.row_eq]
      exact HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup savedRow)
        (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup rfl) (HeapCalls.Evaluates.lookup savedDi) rfl) rfl
    have column : HeapCalls.Evaluates program indexed productColumn heap (.scalar (.int (direction.column c position))) heap := by
      rw [Direction.column_eq]
      exact HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup savedColumn)
        (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup rfl) (HeapCalls.Evaluates.lookup savedDj) rfl) rfl
    have rowBounds : 0 ≤ direction.row r position ∧ direction.row r position < grid.length := by
      rw [Direction.row_eq]
      exact segment_bounds r direction.rowStep grid.length n position (by have := valid.1; constructor <;> omega) valid.2.2.1 active
    have columnBounds : 0 ≤ direction.column c position ∧ direction.column c position < columns := by
      rw [Direction.column_eq]
      exact segment_bounds c direction.columnStep columns n position (by have := valid.2.1; constructor <;> omega) valid.2.2.2 active
    have rowSize : (grid[(direction.row r position).toNat]'(by omega)).length = columns :=
      rectangle _ (List.getElem_mem _)
    let value := ProjectEulerStatements.Grid.get grid (direction.row r position) (direction.column c position) 0
    have read := grid_read heap address grid view indexed _ _ productRow productColumn
      (by intros a b c impossible; cases impossible) (by intros a b c impossible; cases impossible) savedGrid row column
      rowBounds (by simpa only [rowSize] using columnBounds)
    let nextEnv := HeapCalls.setLocal indexed "prod" (.scalar (.int (acc * value)))
    have step : HeapCalls.BlockContinues program indexed productBody heap nextEnv heap :=
      HeapCalls.BlockContinues.augAssign savedAcc read rfl
    obtain ⟨finalEnv, rest, result, preserved⟩ := product.loop direction heap address grid view columns r c n
      (position + 1) rectangle valid nextEnv (acc * value)
      (by simpa [nextEnv, HeapCalls.setLocal] using savedGrid)
      (by simpa [nextEnv, HeapCalls.setLocal] using savedRow)
      (by simpa [nextEnv, HeapCalls.setLocal] using savedColumn)
      (by simpa [nextEnv, HeapCalls.setLocal] using savedDi)
      (by simpa [nextEnv, HeapCalls.setLocal] using savedDj) rfl
    refine ⟨finalEnv, ?_, ?_, ?_⟩
    · apply HeapCalls.BlockContinues.range_step (by simp [Calls.rangeContinues, active])
      simpa only [Nat.cast_add, Nat.cast_one] using step.append rest
    · rw [segmentTail_step grid direction r c position n active]
      simpa only [Nat.cast_mul, Int.mul_assoc] using result
    · intro name notIndex notProduct
      rw [preserved name notIndex notProduct]
      simp [nextEnv, indexed, HeapCalls.setLocal, notIndex, notProduct]
  · refine ⟨env, HeapCalls.BlockContinues.range_done
      (by simp [Calls.rangeContinues, active]) HeapCalls.BlockContinues.nil, ?_, by intros; rfl⟩
    simpa [segmentTail_done grid direction r c position n (by omega)] using getAcc
termination_by n - position
decreasing_by omega

private abbrev updateBest : List Stmt :=
  [.ifThen (.compare (.name "prod") [(.gt, .name "best")])
    [.assign [.name "best"] (.name "prod")] []]

private theorem update_best (heap : Mutable.Heap) (env : HeapCalls.Env) (best value : Nat)
    (getBest : env "best" = some (.scalar (.int best)))
    (getValue : env "prod" = some (.scalar (.int value))) :
    ∃ finalEnv, HeapCalls.BlockContinues program env updateBest heap finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int (↑(max best value)))) ∧
      (∀ name, name ≠ "best" → finalEnv name = env name) := by
  have compared : HeapCalls.Evaluates program env (.compare (.name "prod") [(.gt, .name "best")]) heap
      (.scalar (.bool (decide (best < value)))) heap := by
    apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup getValue) (HeapCalls.Evaluates.lookup getBest)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases larger : best < value
  · refine ⟨HeapCalls.setLocal env "best" (.scalar (.int value)), ?_, ?_, ?_⟩
    · exact HeapCalls.BlockContinues.branch (flag := true) (by simpa [larger] using compared) rfl
        (HeapCalls.BlockContinues.assign (HeapCalls.Evaluates.lookup getValue))
    · simp [HeapCalls.setLocal, max_eq_right (Nat.le_of_lt larger)]
    · intro name other
      simp [HeapCalls.setLocal, other]
  · refine ⟨env, ?_, ?_, by intros; rfl⟩
    · exact HeapCalls.BlockContinues.branch (flag := false) (by simpa [larger] using compared) rfl
        HeapCalls.BlockContinues.nil
    · simpa [max_eq_left (by omega : value ≤ best)] using getBest

private abbrev candidateProduct : List Stmt :=
  [.assign [.name "prod"] (.constant (.int 1)),
    .forIn (.name "t") (.call (.name "range") [.name "k"]) productBody []] ++ updateBest

private theorem product.result (direction : Direction) (heap : Mutable.Heap) (address : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid) (columns r c n : Nat)
    (rectangle : ∀ row ∈ grid, row.length = columns)
    (valid : ValidStart direction grid.length columns n r c) (env : HeapCalls.Env) (best : Nat)
    (getGrid : env "grid" = some (.ref address))
    (getRow : env "i" = some (.scalar (.int r)))
    (getColumn : env "j" = some (.scalar (.int c)))
    (getDi : env "di" = some (.scalar (.int direction.rowStep)))
    (getDj : env "dj" = some (.scalar (.int direction.columnStep)))
    (getCount : env "k" = some (.scalar (.int n)))
    (getBest : env "best" = some (.scalar (.int best))) :
    ∃ finalEnv, HeapCalls.BlockContinues program env candidateProduct heap finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int (↑(max best (segmentProduct grid direction r c n))))) ∧
      (∀ name, name ≠ "t" → name ≠ "prod" → name ≠ "best" → finalEnv name = env name) := by
  let initial := HeapCalls.setLocal env "prod" (.scalar (.int 1))
  have assign : HeapCalls.BlockContinues program env [.assign [.name "prod"] (.constant (.int 1))] heap initial heap :=
    HeapCalls.BlockContinues.assign HeapCalls.Evaluates.constant
  obtain ⟨middle, loop, value, preserved⟩ := product.loop direction heap address grid view columns r c n 0
    rectangle valid initial 1
    (by simpa [initial, HeapCalls.setLocal] using getGrid)
    (by simpa [initial, HeapCalls.setLocal] using getRow)
    (by simpa [initial, HeapCalls.setLocal] using getColumn)
    (by simpa [initial, HeapCalls.setLocal] using getDi)
    (by simpa [initial, HeapCalls.setLocal] using getDj) rfl
  have ran : HeapCalls.BlockContinues program initial
      [.forIn (.name "t") (.call (.name "range") [.name "k"]) productBody []] heap middle heap :=
    HeapCalls.BlockContinues.for_range
      (.cons (HeapCalls.Evaluates.lookup (by simpa [initial, HeapCalls.setLocal] using getCount)) .nil) rfl loop
  obtain ⟨finalEnv, updated, result, saved⟩ := update_best heap middle best (segmentProduct grid direction r c n)
    (by rw [preserved "best" (by decide) (by decide)]; simpa [initial, HeapCalls.setLocal] using getBest)
    (by simpa [segmentTail_zero] using value)
  refine ⟨finalEnv, assign.append (ran.append updated), result, ?_⟩
  intro name notIndex notProduct notBest
  rw [saved name notBest, preserved name notIndex notProduct]
  simp [initial, HeapCalls.setLocal, notProduct]

private def bounded (index length : Expr) : Expr :=
  .compare (.constant (.int 0)) [(.ltE, index), (.lt, length)]

private theorem bounded.result (env : HeapCalls.Env) (heap : Mutable.Heap)
    (index length : Expr) (value : Int) (size : Nat)
    (getIndex : HeapCalls.Evaluates program env index heap (.scalar (.int value)) heap)
    (getLength : HeapCalls.Evaluates program env length heap (.scalar (.int size)) heap) :
    HeapCalls.Evaluates program env (bounded index length) heap
      (.scalar (.bool (decide (0 ≤ value ∧ value < size)))) heap := by
  by_cases nonnegative : 0 ≤ value
  · simp only [nonnegative, true_and]
    apply HeapCalls.Evaluates.chain HeapCalls.Evaluates.constant
    apply HeapCalls.Evaluates.Compares.next (compared := .scalar (.bool true)) (afterComparison := heap) getIndex
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, nonnegative]
    · rfl
    · exact HeapCalls.Evaluates.Compares.last getLength (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  · simp only [nonnegative, false_and, decide_false]
    apply HeapCalls.Evaluates.chain HeapCalls.Evaluates.constant
    exact HeapCalls.Evaluates.Compares.stop getIndex
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, nonnegative]) rfl

private abbrev endpointCondition : Expr :=
  .boolOp .and [bounded (.name "ni") (.name "n"), bounded (.name "nj") (.name "m")]

private theorem endpointCondition.result (heap : Mutable.Heap) (env : HeapCalls.Env) (ni nj : Int) (rows columns : Nat)
    (getNI : env "ni" = some (.scalar (.int ni))) (getNJ : env "nj" = some (.scalar (.int nj)))
    (getRows : env "n" = some (.scalar (.int rows))) (getColumns : env "m" = some (.scalar (.int columns))) :
    HeapCalls.Evaluates program env endpointCondition heap
      (.scalar (.bool (decide ((0 ≤ ni ∧ ni < rows) ∧ (0 ≤ nj ∧ nj < columns))))) heap := by
  have row := bounded.result env heap (.name "ni") (.name "n") ni rows
    (HeapCalls.Evaluates.lookup getNI) (HeapCalls.Evaluates.lookup getRows)
  have column := bounded.result env heap (.name "nj") (.name "m") nj columns
    (HeapCalls.Evaluates.lookup getNJ) (HeapCalls.Evaluates.lookup getColumns)
  by_cases validRow : 0 ≤ ni ∧ ni < rows
  · simp only [validRow, true_and]
    exact HeapCalls.Evaluates.boolean_next (flag := true) (by simpa [validRow] using row) rfl rfl
      (HeapCalls.Evaluates.boolean_last column rfl)
  · simp only [validRow, false_and, decide_false]
    exact HeapCalls.Evaluates.boolean_stop (flag := false) (by simpa [validRow] using row) rfl rfl

private abbrev endpointRow : Expr :=
  .binOp (.name "i") .add (.binOp (.binOp (.name "k") .sub (.constant (.int 1))) .mult (.name "di"))
private abbrev endpointColumn : Expr :=
  .binOp (.name "j") .add (.binOp (.binOp (.name "k") .sub (.constant (.int 1))) .mult (.name "dj"))
private abbrev directionBody : List Stmt :=
  [.assign [.name "ni"] endpointRow, .assign [.name "nj"] endpointColumn,
    .ifThen endpointCondition candidateProduct []]

private theorem direction_result (direction : Direction) (heap : Mutable.Heap) (address : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid) (columns r c n : Nat)
    (rectangle : ∀ row ∈ grid, row.length = columns) (insideRow : r < grid.length) (insideColumn : c < columns)
    (env : HeapCalls.Env) (best : Nat)
    (getGrid : env "grid" = some (.ref address))
    (getRow : env "i" = some (.scalar (.int r)))
    (getColumn : env "j" = some (.scalar (.int c)))
    (getDi : env "di" = some (.scalar (.int direction.rowStep)))
    (getDj : env "dj" = some (.scalar (.int direction.columnStep)))
    (getCount : env "k" = some (.scalar (.int n)))
    (getRows : env "n" = some (.scalar (.int grid.length)))
    (getColumns : env "m" = some (.scalar (.int columns)))
    (getBest : env "best" = some (.scalar (.int best))) :
    ∃ finalEnv, HeapCalls.BlockContinues program env directionBody heap finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int (↑(if ValidStart direction grid.length columns n r c then
        max best (segmentProduct grid direction r c n) else best)))) ∧
      (∀ name, name ∉ ["ni", "nj", "t", "prod", "best"] → finalEnv name = env name) := by
  let ni : Int := r + (n - 1 : Int) * direction.rowStep
  let nj : Int := c + (n - 1 : Int) * direction.columnStep
  let rowEnv := HeapCalls.setLocal env "ni" (.scalar (.int ni))
  let endEnv := HeapCalls.setLocal rowEnv "nj" (.scalar (.int nj))
  have rowAssigned : HeapCalls.BlockContinues program env [.assign [.name "ni"] endpointRow] heap rowEnv heap :=
    HeapCalls.BlockContinues.assign (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup getRow)
      (HeapCalls.Evaluates.binary
        (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup getCount) HeapCalls.Evaluates.constant rfl)
        (HeapCalls.Evaluates.lookup getDi) rfl) rfl)
  have columnAssigned : HeapCalls.BlockContinues program rowEnv [.assign [.name "nj"] endpointColumn] heap endEnv heap :=
    HeapCalls.BlockContinues.assign (HeapCalls.Evaluates.binary
      (HeapCalls.Evaluates.lookup (by simpa [rowEnv, HeapCalls.setLocal] using getColumn))
      (HeapCalls.Evaluates.binary
        (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup (by simpa [rowEnv, HeapCalls.setLocal] using getCount))
          HeapCalls.Evaluates.constant rfl)
        (HeapCalls.Evaluates.lookup (by simpa [rowEnv, HeapCalls.setLocal] using getDj)) rfl) rfl)
  have tested : HeapCalls.Evaluates program endEnv endpointCondition heap
      (.scalar (.bool (decide (ValidStart direction grid.length columns n r c)))) heap := by
    have test := endpointCondition.result heap endEnv ni nj grid.length columns
      (by simp [endEnv, rowEnv, HeapCalls.setLocal]) rfl
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getRows)
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getColumns)
    simpa only [ValidStart, insideRow, insideColumn, true_and] using test
  by_cases valid : ValidStart direction grid.length columns n r c
  · obtain ⟨finalEnv, product, result, preserved⟩ := product.result direction heap address grid view columns r c n
      rectangle valid endEnv best
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getGrid)
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getRow)
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getColumn)
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getDi)
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getDj)
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getCount)
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getBest)
    have branch := HeapCalls.BlockContinues.branch (yes := candidateProduct) (no := []) (flag := true)
      (by simpa [valid] using tested) rfl product
    refine ⟨finalEnv, rowAssigned.append (columnAssigned.append branch), by simpa [valid] using result, ?_⟩
    intro name untouched
    simp only [List.mem_cons, not_or] at untouched
    rw [preserved name untouched.2.2.1 untouched.2.2.2.1 untouched.2.2.2.2.1]
    simp [endEnv, rowEnv, HeapCalls.setLocal, untouched.1, untouched.2.1]
  · have branch := HeapCalls.BlockContinues.branch (yes := candidateProduct) (no := []) (flag := false)
      (by simpa [valid] using tested) rfl HeapCalls.BlockContinues.nil
    refine ⟨endEnv, rowAssigned.append (columnAssigned.append branch), ?_, ?_⟩
    · simpa [valid, endEnv, rowEnv, HeapCalls.setLocal] using getBest
    · intro name untouched
      simp only [List.mem_cons, not_or] at untouched
      simp [endEnv, rowEnv, HeapCalls.setLocal, untouched.1, untouched.2.1]

private abbrev countEnv (k : Int) : HeapCalls.Env :=
  fun name => if name = "k" then some (.scalar (.int k)) else none

private theorem count_condition (heap : Mutable.Heap) (k : Int) :
    HeapCalls.Evaluates program (countEnv k) Solvers.P11Ast.requires_max_adjacent_product_0 heap
      (.scalar (.bool (decide (0 ≤ k)))) heap :=
  HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
    (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])

theorem requires_max_adjacent_product_0_supported (heap : Mutable.Heap) (grid : Nat) (k : Int)
    (_valid : Mutable.Heap.Valid heap) (_typed : Mutable.Value.HasType heap (.ref grid) "list[list[int]]") :
    (HeapCalls.evaluate program (countEnv k) Solvers.P11Ast.requires_max_adjacent_product_0 heap).Supported :=
  Execution.supported_of_finishes (count_condition heap k).finishes

/-- The four direction pairs are Python tuples stored in a Python list. -/
structure DirectionView (heap : Mutable.Heap) (address : Nat) (choices : List Direction) where
  pairs : List Nat
  found : heap[address]? = some (.list (pairs.map Mutable.Value.ref))
  size : pairs.length = choices.length
  contents : ∀ i (within : i < pairs.length), heap[pairs[i]]? = some (.tuple
    [.scalar (.int (choices[i]'(by omega)).rowStep), .scalar (.int (choices[i]'(by omega)).columnStep)])

private theorem directions.loop (heap : Mutable.Heap) (address table : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid) (choices : List Direction)
    (tableView : DirectionView heap table choices) (columns r c n position : Nat)
    (rectangle : ∀ row ∈ grid, row.length = columns) (insideRow : r < grid.length) (insideColumn : c < columns)
    (env : HeapCalls.Env) (best : Nat)
    (getGrid : env "grid" = some (.ref address))
    (getRow : env "i" = some (.scalar (.int r)))
    (getColumn : env "j" = some (.scalar (.int c)))
    (getCount : env "k" = some (.scalar (.int n)))
    (getRows : env "n" = some (.scalar (.int grid.length)))
    (getColumns : env "m" = some (.scalar (.int columns)))
    (getBest : env "best" = some (.scalar (.int best))) :
    ∃ finalEnv, HeapCalls.BlockContinues program env
      [.unpackStepLoop ["di", "dj"] (.ref table) position directionBody []] heap finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int (↑((choices.drop position).foldl
        (scanDirection grid grid.length columns n r c) best)))) ∧
      (∀ name, name ∉ ["di", "dj", "ni", "nj", "t", "prod", "best"] → finalEnv name = env name) := by
  have sequence : Mutable.sequence (.ref table) heap = some (.ok (tableView.pairs.map Mutable.Value.ref), heap) := by
    simp [Mutable.sequence, tableView.found]
  by_cases active : position < choices.length
  · have pairIndex : position < tableView.pairs.length := by rw [tableView.size]; exact active
    let direction := choices[position]
    let pair := tableView.pairs[position]
    let fields : List Mutable.Value := [.scalar (.int direction.rowStep), .scalar (.int direction.columnStep)]
    let boundEnv := HeapCalls.assignNames env ["di", "dj"] fields
    have preserved (name : String) (notDi : name ≠ "di") (notDj : name ≠ "dj") : boundEnv name = env name := by
      simp [boundEnv, fields, HeapCalls.assignNames, HeapCalls.setLocal, notDi, notDj]
    obtain ⟨middle, step, result, saved⟩ := direction_result direction heap address grid view columns r c n
      rectangle insideRow insideColumn boundEnv best
      (by simpa [preserved] using getGrid)
      (by simpa [preserved] using getRow)
      (by simpa [preserved] using getColumn)
      (by simp [boundEnv, fields, HeapCalls.assignNames, HeapCalls.setLocal])
      (by simp [boundEnv, fields, HeapCalls.assignNames, HeapCalls.setLocal])
      (by simpa [preserved] using getCount)
      (by simpa [preserved] using getRows)
      (by simpa [preserved] using getColumns)
      (by simpa [preserved] using getBest)
    obtain ⟨finalEnv, rest, finalBest, finalSaved⟩ := directions.loop heap address table grid view choices tableView
      columns r c n (position + 1) rectangle insideRow insideColumn middle
      (scanDirection grid grid.length columns n r c best direction)
      (by rw [saved "grid" (by simp), preserved "grid" (by decide) (by decide)]; exact getGrid)
      (by rw [saved "i" (by simp), preserved "i" (by decide) (by decide)]; exact getRow)
      (by rw [saved "j" (by simp), preserved "j" (by decide) (by decide)]; exact getColumn)
      (by rw [saved "k" (by simp), preserved "k" (by decide) (by decide)]; exact getCount)
      (by rw [saved "n" (by simp), preserved "n" (by decide) (by decide)]; exact getRows)
      (by rw [saved "m" (by simp), preserved "m" (by decide) (by decide)]; exact getColumns) result
    refine ⟨finalEnv, ?_, ?_, ?_⟩
    · apply HeapCalls.BlockContinues.unpack_step (source := .ref table)
        (items := tableView.pairs.map Mutable.Value.ref) (value := .ref pair) (values := fields) sequence
      · simp [List.getElem?_eq_getElem (show position < (tableView.pairs.map Mutable.Value.ref).length by simpa using pairIndex), pair]
      · have found := tableView.contents position pairIndex
        simpa [Mutable.sequence, pair, fields, direction] using
          (show Mutable.sequence (.ref (tableView.pairs[position])) heap = some (.ok fields, heap) from by
            simp [Mutable.sequence, found, fields, direction])
      · rfl
      · simpa [directionBody, Calls.loopContinuation, Calls.loopControl] using step.append rest
    · rw [← List.cons_getElem_drop_succ (h := active), List.foldl_cons]
      exact finalBest
    · intro name untouched
      rw [finalSaved name untouched]
      have notDi : name ≠ "di" := by intro h; exact untouched (by simp [h])
      have notDj : name ≠ "dj" := by intro h; exact untouched (by simp [h])
      have notOther : name ∉ ["ni", "nj", "t", "prod", "best"] := by
        intro member
        exact untouched (by simp only [List.mem_cons] at member ⊢; tauto)
      rw [saved name notOther, preserved name notDi notDj]
  · have outside : choices.length ≤ position := by omega
    refine ⟨env, ?_, ?_, by intros; rfl⟩
    · exact HeapCalls.BlockContinues.unpack_done sequence
        (List.getElem?_eq_none (by simpa [tableView.size] using outside)) HeapCalls.BlockContinues.nil
    · simpa [List.drop_eq_nil_of_le outside] using getBest
termination_by choices.length - position
decreasing_by omega

private abbrev directionLoop : List Stmt :=
  [.forIn (.tuple [.name "di", .name "dj"]) (.name "directions") directionBody []]

private theorem directions.result (heap : Mutable.Heap) (address table : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid)
    (tableView : DirectionView heap table directions) (columns r c n : Nat)
    (rectangle : ∀ row ∈ grid, row.length = columns) (insideRow : r < grid.length) (insideColumn : c < columns)
    (env : HeapCalls.Env) (best : Nat)
    (getGrid : env "grid" = some (.ref address))
    (getRow : env "i" = some (.scalar (.int r)))
    (getColumn : env "j" = some (.scalar (.int c)))
    (getCount : env "k" = some (.scalar (.int n)))
    (getRows : env "n" = some (.scalar (.int grid.length)))
    (getColumns : env "m" = some (.scalar (.int columns)))
    (getBest : env "best" = some (.scalar (.int best)))
    (getTable : env "directions" = some (.ref table)) :
    ∃ finalEnv, HeapCalls.BlockContinues program env directionLoop heap finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int (↑((directions.filterMap fun direction =>
        if ValidStart direction grid.length columns n r c then
          some (segmentProduct grid direction r c n) else none).foldl max best)))) ∧
      (∀ name, name ∉ ["di", "dj", "ni", "nj", "t", "prod", "best"] → finalEnv name = env name) := by
  obtain ⟨finalEnv, loop, result, saved⟩ := directions.loop heap address table grid view directions tableView
    columns r c n 0 rectangle insideRow insideColumn env best getGrid getRow getColumn getCount getRows getColumns getBest
  refine ⟨finalEnv, ?_, ?_, saved⟩
  · exact HeapCalls.BlockContinues.for_unpack (items := tableView.pairs.map Mutable.Value.ref)
      (by intros values impossible; cases impossible) rfl
      (HeapCalls.Evaluates.lookup getTable) (by simp [Mutable.sequence, tableView.found]) loop
  · simpa only [List.drop_zero, fold_scanDirection] using result

private theorem column_loop (heap : Mutable.Heap) (address table : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid)
    (tableView : DirectionView heap table directions) (columns r n position : Nat)
    (rectangle : ∀ row ∈ grid, row.length = columns) (insideRow : r < grid.length)
    (env : HeapCalls.Env) (best : Nat)
    (getGrid : env "grid" = some (.ref address))
    (getRow : env "i" = some (.scalar (.int r)))
    (getCount : env "k" = some (.scalar (.int n)))
    (getRows : env "n" = some (.scalar (.int grid.length)))
    (getColumns : env "m" = some (.scalar (.int columns)))
    (getBest : env "best" = some (.scalar (.int best)))
    (getTable : env "directions" = some (.ref table)) :
    ∃ finalEnv, HeapCalls.BlockContinues program env
      [.rangeStepLoop "j" position columns 1 directionLoop []] heap finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int (↑((scanRowTail grid grid.length columns n r position).foldl max best)))) ∧
      (∀ name, name ∉ ["j", "di", "dj", "ni", "nj", "t", "prod", "best"] → finalEnv name = env name) := by
  by_cases active : position < columns
  · let boundEnv := HeapCalls.setLocal env "j" (.scalar (.int position))
    obtain ⟨middle, step, result, saved⟩ := directions.result heap address table grid view tableView
      columns r position n rectangle insideRow active boundEnv best
      (by simpa [boundEnv, HeapCalls.setLocal] using getGrid)
      (by simpa [boundEnv, HeapCalls.setLocal] using getRow) rfl
      (by simpa [boundEnv, HeapCalls.setLocal] using getCount)
      (by simpa [boundEnv, HeapCalls.setLocal] using getRows)
      (by simpa [boundEnv, HeapCalls.setLocal] using getColumns)
      (by simpa [boundEnv, HeapCalls.setLocal] using getBest)
      (by simpa [boundEnv, HeapCalls.setLocal] using getTable)
    let nextBest := (directions.filterMap fun direction =>
      if ValidStart direction grid.length columns n r position then
        some (segmentProduct grid direction r position n) else none).foldl max best
    obtain ⟨finalEnv, rest, finalBest, finalSaved⟩ := column_loop heap address table grid view tableView
      columns r n (position + 1) rectangle insideRow middle nextBest
      (by rw [saved "grid" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getGrid)
      (by rw [saved "i" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getRow)
      (by rw [saved "k" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getCount)
      (by rw [saved "n" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getRows)
      (by rw [saved "m" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getColumns) result
      (by rw [saved "directions" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getTable)
    refine ⟨finalEnv, ?_, ?_, ?_⟩
    · apply HeapCalls.BlockContinues.range_step (by simp [Calls.rangeContinues, active])
      simpa [directionLoop, Calls.loopContinuation, Calls.loopControl, Nat.cast_add, Nat.cast_one] using step.append rest
    · rw [scanRowTail_step grid grid.length columns n r position active, List.foldl_append]
      exact finalBest
    · intro name untouched
      rw [finalSaved name untouched, saved name (by intro member; exact untouched (List.mem_cons_of_mem _ member))]
      have notColumn : name ≠ "j" := by intro equal; exact untouched (by simp [equal])
      simp [boundEnv, HeapCalls.setLocal, notColumn]
  · refine ⟨env, HeapCalls.BlockContinues.range_done
      (by simp [Calls.rangeContinues, active]) HeapCalls.BlockContinues.nil, ?_, by intros; rfl⟩
    simpa [scanRowTail_done grid grid.length columns n r position (by omega)] using getBest
termination_by columns - position
decreasing_by omega

private abbrev columnLoop : List Stmt :=
  [.forIn (.name "j") (.call (.name "range") [.name "m"]) directionLoop []]

private theorem rows.loop (heap : Mutable.Heap) (address table : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid)
    (tableView : DirectionView heap table directions) (columns n position : Nat)
    (rectangle : ∀ row ∈ grid, row.length = columns) (env : HeapCalls.Env) (best : Nat)
    (getGrid : env "grid" = some (.ref address))
    (getCount : env "k" = some (.scalar (.int n)))
    (getRows : env "n" = some (.scalar (.int grid.length)))
    (getColumns : env "m" = some (.scalar (.int columns)))
    (getBest : env "best" = some (.scalar (.int best)))
    (getTable : env "directions" = some (.ref table)) :
    ∃ finalEnv, HeapCalls.BlockContinues program env
      [.rangeStepLoop "i" position grid.length 1 columnLoop []] heap finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int (↑((scanGridTail grid grid.length columns n position).foldl max best)))) ∧
      (∀ name, name ∉ ["i", "j", "di", "dj", "ni", "nj", "t", "prod", "best"] → finalEnv name = env name) := by
  by_cases active : position < grid.length
  · let boundEnv := HeapCalls.setLocal env "i" (.scalar (.int position))
    obtain ⟨middle, columnsRan, result, saved⟩ := column_loop heap address table grid view tableView
      columns position n 0 rectangle active boundEnv best
      (by simpa [boundEnv, HeapCalls.setLocal] using getGrid) rfl
      (by simpa [boundEnv, HeapCalls.setLocal] using getCount)
      (by simpa [boundEnv, HeapCalls.setLocal] using getRows)
      (by simpa [boundEnv, HeapCalls.setLocal] using getColumns)
      (by simpa [boundEnv, HeapCalls.setLocal] using getBest)
      (by simpa [boundEnv, HeapCalls.setLocal] using getTable)
    have step : HeapCalls.BlockContinues program boundEnv columnLoop heap middle heap :=
      HeapCalls.BlockContinues.for_range
        (.cons (HeapCalls.Evaluates.lookup (by simpa [boundEnv, HeapCalls.setLocal] using getColumns)) .nil) rfl columnsRan
    obtain ⟨finalEnv, rest, finalBest, finalSaved⟩ := rows.loop heap address table grid view tableView
      columns n (position + 1) rectangle middle ((scanRowTail grid grid.length columns n position 0).foldl max best)
      (by rw [saved "grid" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getGrid)
      (by rw [saved "k" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getCount)
      (by rw [saved "n" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getRows)
      (by rw [saved "m" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getColumns) result
      (by rw [saved "directions" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getTable)
    refine ⟨finalEnv, ?_, ?_, ?_⟩
    · apply HeapCalls.BlockContinues.range_step (by simp [Calls.rangeContinues, active])
      simpa [columnLoop, Calls.loopContinuation, Calls.loopControl, Nat.cast_add, Nat.cast_one] using step.append rest
    · rw [scanGridTail_step grid grid.length columns n position active, List.foldl_append]
      exact finalBest
    · intro name untouched
      rw [finalSaved name untouched, saved name (by intro member; exact untouched (List.mem_cons_of_mem _ member))]
      have notRow : name ≠ "i" := by intro equal; exact untouched (by simp [equal])
      simp [boundEnv, HeapCalls.setLocal, notRow]
  · refine ⟨env, HeapCalls.BlockContinues.range_done
      (by simp [Calls.rangeContinues, active]) HeapCalls.BlockContinues.nil, ?_, by intros; rfl⟩
    simpa [scanGridTail_done grid grid.length columns n position (by omega)] using getBest
termination_by grid.length - position
decreasing_by omega

private def pairObject (direction : Direction) : Mutable.Object :=
  .tuple [.scalar (.int direction.rowStep), .scalar (.int direction.columnStep)]

private def directionObjects (heap : Mutable.Heap) : Mutable.Heap :=
  directions.map pairObject ++ [.list ([heap.length, heap.length + 1, heap.length + 2, heap.length + 3].map Mutable.Value.ref)]

private def directionView (heap : Mutable.Heap) :
    DirectionView (heap ++ directionObjects heap) (heap.length + 4) directions := by
  refine ⟨[heap.length, heap.length + 1, heap.length + 2, heap.length + 3], ?_, rfl, ?_⟩
  · simp [directionObjects, directions]
  · intro i within
    have bound : i < 4 := within
    interval_cases i <;> simp [directionObjects, directions, pairObject]

private abbrev directionExpression : Expr := .list
  [.tuple [.constant (.int 0), .constant (.int 1)],
    .tuple [.constant (.int 1), .constant (.int 0)],
    .tuple [.constant (.int 1), .constant (.int 1)],
    .tuple [.constant (.int 1), .unaryOp .uSub (.constant (.int 1))]]

private theorem directions.allocate (heap : Mutable.Heap) (env : HeapCalls.Env) :
    HeapCalls.Evaluates program env directionExpression heap (.ref (heap.length + 4)) (heap ++ directionObjects heap) := by
  have first : HeapCalls.Evaluates program env (.tuple [.constant (.int 0), .constant (.int 1)]) heap
      (.ref heap.length) (heap ++ [pairObject .right]) :=
    HeapCalls.Evaluates.tuple (.cons HeapCalls.Evaluates.constant (.cons HeapCalls.Evaluates.constant .nil)) (by simp)
  have second : HeapCalls.Evaluates program env (.tuple [.constant (.int 1), .constant (.int 0)])
      (heap ++ [pairObject .right]) (.ref (heap.length + 1)) (heap ++ [pairObject .right, pairObject .down]) := by
    simpa [List.append_assoc] using (HeapCalls.Evaluates.tuple (module := program) (env := env)
      (heap := heap ++ [pairObject .right])
      (.cons HeapCalls.Evaluates.constant (.cons HeapCalls.Evaluates.constant .nil))
      (by simp : ([.scalar (.int 1), .scalar (.int 0)] : List Mutable.Value) ≠ []))
  have third : HeapCalls.Evaluates program env (.tuple [.constant (.int 1), .constant (.int 1)])
      (heap ++ [pairObject .right, pairObject .down]) (.ref (heap.length + 2))
      (heap ++ [pairObject .right, pairObject .down, pairObject .downRight]) := by
    simpa [List.append_assoc] using (HeapCalls.Evaluates.tuple (module := program) (env := env)
      (heap := heap ++ [pairObject .right, pairObject .down])
      (.cons HeapCalls.Evaluates.constant (.cons HeapCalls.Evaluates.constant .nil))
      (by simp : ([.scalar (.int 1), .scalar (.int 1)] : List Mutable.Value) ≠ []))
  have fourth : HeapCalls.Evaluates program env (.tuple [.constant (.int 1), .unaryOp .uSub (.constant (.int 1))])
      (heap ++ [pairObject .right, pairObject .down, pairObject .downRight]) (.ref (heap.length + 3))
      (heap ++ [pairObject .right, pairObject .down, pairObject .downRight, pairObject .downLeft]) := by
    simpa [List.append_assoc] using (HeapCalls.Evaluates.tuple (module := program) (env := env)
      (heap := heap ++ [pairObject .right, pairObject .down, pairObject .downRight])
      (.cons HeapCalls.Evaluates.constant (.cons (HeapCalls.Evaluates.unary (op := .uSub) (value := .scalar (.int 1)) HeapCalls.Evaluates.constant rfl) .nil))
      (by simp : ([.scalar (.int 1), .scalar (.int (-1))] : List Mutable.Value) ≠ []))
  simpa [directionObjects, directions, List.append_assoc] using
    HeapCalls.Evaluates.list (.cons first (.cons second (.cons third (.cons fourth .nil))))

private abbrev directionAnnotation : Expr :=
  .subscript (.name "list") (.subscript (.name "tuple") (.tuple [.name "int", .name "int"]))

/-- The optimized source returns the cell-first mathematical maximum, extending
only the private heap with its direction table. -/
theorem max_adjacent_product.exact (heap : Mutable.Heap) (address : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid) (columns n : Nat)
    (nonempty : 0 < grid.length) (rectangle : ∀ row ∈ grid, row.length = columns) :
    HeapCalls.ReturnsValue program "max_adjacent_product" [.ref address, .scalar (.int n)] heap
      (.scalar (.int (↑((scannedProducts grid grid.length columns n).foldl max 0))))
      (heap ++ directionObjects heap) := by
  let args : List Arg :=
    [⟨"grid", some (.subscript (.name "list") (.subscript (.name "list") (.name "int"))), none⟩,
      ⟨"k", some (.name "int"), some (.int 4)⟩]
  let initial := HeapCalls.argumentEnv args [.ref address, .scalar (.int n)]
  let withRows := HeapCalls.setLocal initial "n" (.scalar (.int grid.length))
  let withColumns := HeapCalls.setLocal withRows "m" (.scalar (.int columns))
  let withTable := HeapCalls.setLocal withColumns "directions" (.ref (heap.length + 4))
  let withBest := HeapCalls.setLocal withTable "best" (.scalar (.int 0))
  have getLength : HeapCalls.Evaluates program initial (.builtinCall .len [.name "grid"]) heap
      (.scalar (.int grid.length)) heap := by
    exact HeapCalls.Evaluates.builtin (.cons (HeapCalls.Evaluates.lookup rfl) .nil)
      (by simp [HeapCalls.primitive, Mutable.length, view.found, view.size])
  have firstIndex : 0 < view.rows.length := by rw [view.size]; exact nonempty
  have rowSize : (grid[0]'nonempty).length = columns := rectangle _ (List.getElem_mem nonempty)
  have getWidth : HeapCalls.Evaluates program withRows
      (.ifExp (.name "n") (.builtinCall .len [.subscript (.name "grid") (.constant (.int 0))]) (.constant (.int 0))) heap
      (.scalar (.int columns)) heap := by
    apply HeapCalls.Evaluates.conditional (flag := true) (HeapCalls.Evaluates.lookup rfl)
    · simp [Mutable.truth, scalarTruth, nonempty.ne']
    · have readRow : HeapCalls.Evaluates program withRows
          (.subscript (.name "grid") (.constant (.int 0))) heap (.ref (view.rows[0])) heap := by
        apply HeapCalls.Evaluates.subscript (by intros a b c impossible; cases impossible)
          (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
        have read := Mutable.getItem_list heap address 0 (view.rows.map Mutable.Value.ref) view.found (by simpa using firstIndex)
        rw [List.getElem_map] at read
        exact read
      apply HeapCalls.Evaluates.builtin (.cons readRow .nil)
      simp [HeapCalls.primitive, Mutable.length, view.contents 0 firstIndex, rowSize]

  obtain ⟨finalEnv, scan, result, _⟩ := rows.loop (heap ++ directionObjects heap) address (heap.length + 4)
    grid (view.append _) (directionView heap) columns n 0 rectangle withBest 0 rfl rfl rfl rfl rfl rfl
  have loop : HeapCalls.BlockContinues program withBest
      [.forIn (.name "i") (.call (.name "range") [.name "n"]) columnLoop []]
      (heap ++ directionObjects heap) finalEnv (heap ++ directionObjects heap) :=
    HeapCalls.BlockContinues.for_range (.cons (HeapCalls.Evaluates.lookup rfl) .nil) rfl scan
  apply HeapCalls.BlockReturns.function (args := args)
    (body := [.expr (.constant (.string "\n    @formalsw: Py from project_euler_statements.p11 import naive\n    @formalsw: Requires(k >= 0)\n    @formalsw: Requires(is_euler_grid(grid))\n    @formalsw: Equiv(naive)\n    ")),
      .assign [.name "n"] (.builtinCall .len [.name "grid"]),
      .assign [.name "m"] (.ifExp (.name "n") (.builtinCall .len [.subscript (.name "grid") (.constant (.int 0))]) (.constant (.int 0))),
      .annAssign (.name "directions") directionAnnotation (some directionExpression),
      .assign [.name "best"] (.constant (.int 0)),
      .forIn (.name "i") (.call (.name "range") [.name "n"]) columnLoop [], .return (.name "best")]) rfl rfl
  apply HeapCalls.BlockReturns.assign getLength
  apply HeapCalls.BlockReturns.assign getWidth
  apply HeapCalls.BlockContinues.returns (HeapCalls.BlockContinues.annAssign (directions.allocate heap withColumns))
  apply HeapCalls.BlockReturns.assign HeapCalls.Evaluates.constant
  exact loop.returns (HeapCalls.BlockReturns.return_value (HeapCalls.Evaluates.lookup
    (by simpa [scanGridTail_zero] using result)))

private abbrev intItems (values : List Int) : List Mutable.Value :=
  values.map (fun value => .scalar (.int value))

private def rowValues (heap : Mutable.Heap) (address : Nat) : List Int :=
  match heap[address]? with
  | some (.list values) => values.map fun value => match value with
      | .scalar (.int n) => n
      | _ => 0
  | _ => []

private theorem rowValues.found (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[int]") :
    heap[address]? = some (.list (intItems (rowValues heap address))) := by
  obtain ⟨values, found⟩ := Mutable.Value.ints_of_hasType typed
  simp [rowValues, found, intItems, List.map_map]

private def ValuesGood (values : List Int) : Prop := ∀ value ∈ values, 0 ≤ value
private def RowGood (heap : Mutable.Heap) (address : Nat) : Prop :=
  (rowValues heap address).length = 20 ∧ ValuesGood (rowValues heap address)

private abbrev valueCheckBody : List Stmt :=
  [.ifThen (.compare (.name "value") [(.lt, .constant (.int 0))]) [.return (.constant (.bool false))] []]

private theorem values_check (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.list (intItems values))) (position : Nat) (env : HeapCalls.Env) :
    (ValuesGood (values.drop position) ∧ ∃ finalEnv, HeapCalls.BlockContinues program env
      [.sequenceStepLoop "value" (.ref address) position valueCheckBody []] heap finalEnv heap ∧
      (∀ name, name ≠ "value" → finalEnv name = env name)) ∨
    (∀ following, HeapCalls.BlockReturns program env
      (.sequenceStepLoop "value" (.ref address) position valueCheckBody [] :: following) heap (.scalar (.bool false)) heap) := by
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (intItems values), heap) := by
    simp [Mutable.sequence, found]
  by_cases active : position < values.length
  · let value := values[position]
    let boundEnv := HeapCalls.setLocal env "value" (.scalar (.int value))
    have item : (intItems values)[position]? = some (.scalar (.int value)) := by
      simp [intItems, List.getElem?_eq_getElem active, value]
    have tested : HeapCalls.Evaluates program boundEnv (.compare (.name "value") [(.lt, .constant (.int 0))]) heap
        (.scalar (.bool (decide (value < 0)))) heap :=
      HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
    by_cases negative : value < 0
    · right
      intro following
      apply HeapCalls.BlockReturns.sequence_step (source := .ref address) sequence item
      exact HeapCalls.BlockReturns.branch (flag := true) (by simpa [negative] using tested) rfl
        (HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant)
    · have step : HeapCalls.BlockContinues program boundEnv valueCheckBody heap boundEnv heap :=
        HeapCalls.BlockContinues.branch (flag := false) (by simpa [negative] using tested) rfl HeapCalls.BlockContinues.nil
      rcases values_check heap address values found (position + 1) boundEnv with ⟨good, finalEnv, rest, saved⟩ | rejected
      · left
        refine ⟨?_, finalEnv, ?_, ?_⟩
        · rw [List.drop_eq_getElem_cons active]
          simpa only [ValuesGood, List.forall_mem_cons] using And.intro (by omega : 0 ≤ value) good
        · apply HeapCalls.BlockContinues.sequence_step (source := .ref address) sequence item
          simpa [valueCheckBody, Calls.loopContinuation, Calls.loopControl] using step.append rest
        · intro name other
          rw [saved name other]
          simp [boundEnv, HeapCalls.setLocal, other]
      · right
        intro following
        apply HeapCalls.BlockReturns.sequence_step (source := .ref address) sequence item
        simpa [valueCheckBody, Calls.loopContinuation, Calls.loopControl] using step.returns (rejected following)
  · left
    refine ⟨by simp [ValuesGood, List.drop_eq_nil_of_le (by omega : values.length ≤ position)], env, ?_, by intros; rfl⟩
    exact HeapCalls.BlockContinues.sequence_done sequence
      (List.getElem?_eq_none (by simpa [intItems] using (show values.length ≤ position by omega))) HeapCalls.BlockContinues.nil
termination_by values.length - position
decreasing_by omega

private abbrev rowLengthTest : Expr :=
  .compare (.builtinCall .len [.name "row"]) [(.notEq, .constant (.int 20))]
private abbrev rowCheckBody : List Stmt :=
  [.ifThen rowLengthTest [.return (.constant (.bool false))] [],
    .forIn (.name "value") (.name "row") valueCheckBody []]

private theorem row_check (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[int]") (env : HeapCalls.Env)
    (getRow : env "row" = some (.ref address)) :
    (RowGood heap address ∧ ∃ finalEnv, HeapCalls.BlockContinues program env rowCheckBody heap finalEnv heap ∧
      (∀ name, name ≠ "value" → finalEnv name = env name)) ∨
    (∀ following, HeapCalls.BlockReturns program env (rowCheckBody ++ following) heap (.scalar (.bool false)) heap) := by
  let values := rowValues heap address
  have found := rowValues.found heap address typed
  have tested : HeapCalls.Evaluates program env rowLengthTest heap
      (.scalar (.bool (decide (values.length ≠ 20)))) heap := by
    have length : HeapCalls.Evaluates program env (.builtinCall .len [.name "row"]) heap
        (.scalar (.int values.length)) heap :=
      HeapCalls.Evaluates.builtin (.cons (HeapCalls.Evaluates.lookup getRow) .nil)
        (by simp [HeapCalls.primitive, Mutable.length, found, intItems, values])
    exact HeapCalls.Evaluates.compare length HeapCalls.Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (intItems values), heap) := by
    simp [Mutable.sequence, found, values]
  by_cases lengthGood : values.length = 20
  · have skipped : HeapCalls.BlockContinues program env
        [.ifThen rowLengthTest [.return (.constant (.bool false))] []] heap env heap :=
      HeapCalls.BlockContinues.branch (flag := false) (by simpa [lengthGood] using tested) rfl HeapCalls.BlockContinues.nil
    rcases values_check heap address values found 0 env with ⟨good, finalEnv, loop, saved⟩ | rejected
    · left
      refine ⟨⟨lengthGood, by simpa using good⟩, finalEnv, ?_, saved⟩
      have entered : HeapCalls.BlockContinues program env
          [.forIn (.name "value") (.name "row") valueCheckBody []] heap finalEnv heap :=
        HeapCalls.BlockContinues.for_sequence (by intros args impossible; cases impossible)
          (HeapCalls.Evaluates.lookup getRow) sequence loop
      exact skipped.append entered
    · right
      intro following
      apply skipped.returns
      exact HeapCalls.BlockReturns.for_sequence (by intros args impossible; cases impossible)
        (HeapCalls.Evaluates.lookup getRow) sequence (rejected following)
  · right
    intro following
    exact HeapCalls.BlockReturns.branch (flag := true) (by simpa [lengthGood] using tested) rfl
      (HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant)

private theorem rows_check (heap : Mutable.Heap) (address : Nat) (rows : List Nat)
    (found : heap[address]? = some (.list (rows.map Mutable.Value.ref)))
    (typed : ∀ row ∈ rows, Mutable.Value.HasType heap (.ref row) "list[int]") (position : Nat) (env : HeapCalls.Env) :
    ((∀ row ∈ rows.drop position, RowGood heap row) ∧ ∃ finalEnv, HeapCalls.BlockContinues program env
      [.sequenceStepLoop "row" (.ref address) position rowCheckBody []] heap finalEnv heap ∧
      (∀ name, name ≠ "row" → name ≠ "value" → finalEnv name = env name)) ∨
    (∀ following, HeapCalls.BlockReturns program env
      (.sequenceStepLoop "row" (.ref address) position rowCheckBody [] :: following) heap (.scalar (.bool false)) heap) := by
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (rows.map Mutable.Value.ref), heap) := by
    simp [Mutable.sequence, found]
  by_cases active : position < rows.length
  · let row := rows[position]
    let boundEnv := HeapCalls.setLocal env "row" (.ref row)
    have item : (rows.map Mutable.Value.ref)[position]? = some (.ref row) := by
      simp [List.getElem?_eq_getElem active, row]
    rcases row_check heap row (typed row (List.getElem_mem active)) boundEnv rfl with
      ⟨goodRow, middle, step, saved⟩ | rejectedRow
    · rcases rows_check heap address rows found typed (position + 1) middle with
        ⟨goodRows, finalEnv, rest, finalSaved⟩ | rejected
      · left
        refine ⟨?_, finalEnv, ?_, ?_⟩
        · rw [List.drop_eq_getElem_cons active]
          simpa only [List.forall_mem_cons] using And.intro goodRow goodRows
        · apply HeapCalls.BlockContinues.sequence_step (source := .ref address) sequence item
          simpa [rowCheckBody, Calls.loopContinuation, Calls.loopControl] using step.append rest
        · intro name notRow notValue
          rw [finalSaved name notRow notValue, saved name notValue]
          simp [boundEnv, HeapCalls.setLocal, notRow]
      · right
        intro following
        apply HeapCalls.BlockReturns.sequence_step (source := .ref address) sequence item
        simpa [rowCheckBody, Calls.loopContinuation, Calls.loopControl] using step.returns (rejected following)
    · right
      intro following
      apply HeapCalls.BlockReturns.sequence_step (source := .ref address) sequence item
      exact rejectedRow _
  · left
    refine ⟨by simp [List.drop_eq_nil_of_le (by omega : rows.length ≤ position)], env, ?_, by intros; rfl⟩
    exact HeapCalls.BlockContinues.sequence_done sequence
      (List.getElem?_eq_none (by simpa using (show rows.length ≤ position by omega))) HeapCalls.BlockContinues.nil
termination_by rows.length - position
decreasing_by omega

private def GridDomain (heap : Mutable.Heap) (address : Nat) : Prop :=
  ∃ grid : List (List Nat), Nonempty (GridView heap address grid) ∧ grid.length = 20 ∧
    ∀ row ∈ grid, row.length = 20

private theorem good_grid_domain (heap : Mutable.Heap) (address : Nat) (rows : List Nat)
    (found : heap[address]? = some (.list (rows.map Mutable.Value.ref)))
    (typed : ∀ row ∈ rows, Mutable.Value.HasType heap (.ref row) "list[int]")
    (size : rows.length = 20) (good : ∀ row ∈ rows, RowGood heap row) : GridDomain heap address := by
  let grid := rows.map fun row => (rowValues heap row).map Int.toNat
  have encoded (row : Nat) (member : row ∈ rows) :
      intItems (rowValues heap row) =
        ((rowValues heap row).map Int.toNat).map (fun (value : Nat) => Mutable.Value.scalar (.int value)) := by
    simp only [intItems, List.map_map]
    apply List.map_congr_left
    intro value present
    have nonnegative := (good row member).2 value present
    simp [Int.toNat_of_nonneg nonnegative]
  have view : GridView heap address grid := by
    refine ⟨rows, found, by simp [grid], ?_⟩
    intro i within
    have rowFound := rowValues.found heap rows[i] (typed rows[i] (List.getElem_mem within))
    rw [encoded rows[i] (List.getElem_mem within)] at rowFound
    simpa [grid] using rowFound
  refine ⟨grid, ⟨view⟩, by simpa [grid] using size, ?_⟩
  intro row member
  obtain ⟨address, present, rfl⟩ := List.mem_map.mp member
  simpa using (good address present).1

private theorem is_euler_grid.result (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]") :
    (GridDomain heap address ∧ HeapCalls.ReturnsValue program "is_euler_grid" [.ref address] heap (.scalar (.bool true)) heap) ∨
    HeapCalls.ReturnsValue program "is_euler_grid" [.ref address] heap (.scalar (.bool false)) heap := by
  obtain ⟨rows, found, types⟩ := Mutable.Value.lists_of_hasType typed
  let args : List Arg := [⟨"grid", some (.subscript (.name "list") (.subscript (.name "list") (.name "int"))), none⟩]
  let env := HeapCalls.argumentEnv args [.ref address]
  have length : HeapCalls.Evaluates program env (.builtinCall .len [.name "grid"]) heap
      (.scalar (.int rows.length)) heap :=
    HeapCalls.Evaluates.builtin (.cons (HeapCalls.Evaluates.lookup rfl) .nil)
      (by simp [HeapCalls.primitive, Mutable.length, found])
  have tested : HeapCalls.Evaluates program env
      (.compare (.builtinCall .len [.name "grid"]) [(.notEq, .constant (.int 20))]) heap
      (.scalar (.bool (decide (rows.length ≠ 20)))) heap :=
    HeapCalls.Evaluates.compare length HeapCalls.Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (rows.map Mutable.Value.ref), heap) := by
    simp [Mutable.sequence, found]
  by_cases size : rows.length = 20
  · rcases rows_check heap address rows found types 0 env with ⟨good, finalEnv, loop, _⟩ | rejected
    · left
      refine ⟨good_grid_domain heap address rows found types size (by simpa using good), ?_⟩
      apply HeapCalls.BlockReturns.function (args := args) rfl rfl
      apply HeapCalls.BlockReturns.branch (flag := false) (by simpa [size] using tested) rfl
      have entered : HeapCalls.BlockContinues program env [.forIn (.name "row") (.name "grid") rowCheckBody []]
          heap finalEnv heap :=
        HeapCalls.BlockContinues.for_sequence (by intros expressions impossible; cases impossible)
          (HeapCalls.Evaluates.lookup rfl) sequence loop
      exact entered.returns (HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant)
    · right
      apply HeapCalls.BlockReturns.function (args := args) rfl rfl
      apply HeapCalls.BlockReturns.branch (flag := false) (by simpa [size] using tested) rfl
      exact HeapCalls.BlockReturns.for_sequence (by intros expressions impossible; cases impossible)
        (HeapCalls.Evaluates.lookup rfl) sequence (rejected [.return (.constant (.bool true))])
  · right
    apply HeapCalls.BlockReturns.function (args := args) rfl rfl
    exact HeapCalls.BlockReturns.branch (flag := true) (by simpa [size] using tested) rfl
      (HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant)

private abbrev gridEnv (address : Nat) : HeapCalls.Env :=
  fun name => if name = "grid" then some (.ref address) else none

private theorem grid_condition (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]") :
    ∃ flag, HeapCalls.Evaluates program (gridEnv address) Solvers.P11Ast.requires_max_adjacent_product_1 heap
      (.scalar (.bool flag)) heap ∧ (flag = true → GridDomain heap address) := by
  rcases is_euler_grid.result heap address typed with ⟨domain, executed⟩ | executed
  · exact ⟨true, HeapCalls.Evaluates.call (.cons (HeapCalls.Evaluates.lookup rfl) .nil) executed, by intros; exact domain⟩
  · exact ⟨false, HeapCalls.Evaluates.call (.cons (HeapCalls.Evaluates.lookup rfl) .nil) executed, by simp⟩

private theorem grid_domain_of_condition (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]")
    (requires : HeapCalls.Condition program (gridEnv address) Solvers.P11Ast.requires_max_adjacent_product_1 heap) :
    GridDomain heap address := by
  obtain ⟨flag, evaluated, domain⟩ := grid_condition heap address typed
  exact domain (by simpa using evaluated.condition_iff.mp requires)

theorem requires_max_adjacent_product_1_supported (heap : Mutable.Heap) (grid : Nat) (k : Int)
    (_valid : Mutable.Heap.Valid heap) (typed : Mutable.Value.HasType heap (.ref grid) "list[list[int]]")
    (_count : HeapCalls.Condition program (countEnv k) Solvers.P11Ast.requires_max_adjacent_product_0 heap) :
    (HeapCalls.evaluate program (gridEnv grid) Solvers.P11Ast.requires_max_adjacent_product_1 heap).Supported := by
  obtain ⟨_, evaluated, _⟩ := grid_condition heap grid typed
  exact Execution.supported_of_finishes evaluated.finishes

theorem max_adjacent_product.«Equiv(naive)» (heap : Mutable.Heap) (address : Nat) (k : Int)
    (valid : Mutable.Heap.Valid heap) (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]")
    (count : HeapCalls.Condition program (countEnv k) Solvers.P11Ast.requires_max_adjacent_product_0 heap)
    (domain : HeapCalls.Condition program (gridEnv address) Solvers.P11Ast.requires_max_adjacent_product_1 heap) :
    (HeapCalls.call program "max_adjacent_product" [.ref address, .scalar (.int k)] heap).Equiv
      (HeapCalls.call P11Ast.module "naive" [.ref address, .scalar (.int k)] heap) := by
  have nonnegative : 0 ≤ k := by simpa using (count_condition heap k).condition_iff.mp count
  lift k to Nat using nonnegative
  obtain ⟨grid, ⟨view⟩, size, rectangle⟩ := grid_domain_of_condition heap address typed domain
  have optimized := max_adjacent_product.exact heap address grid view 20 k (by omega) rectangle
  obtain ⟨extra, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p11.naive.exact heap address grid view k
  have same : (scannedProducts grid grid.length 20 k).foldl max 0 = (directionalProducts grid 20 20 k).foldl max 0 := by
    rw [size, ← ProjectEulerStatements.P11.naive_eq_scanned, ProjectEulerStatements.P11.naive_eq_directional]
  rw [same] at optimized
  exact Mutable.Call.equiv_of_finishes optimized.finishes reference.finishes
    (Mutable.Completion.equivalent_allocations heap (directionObjects heap) extra
      (.scalar (.int (↑((directionalProducts grid 20 20 k).foldl max 0)))) [] valid trivial (by simp))

theorem max_adjacent_product.«Supported» :
    (∀ (heap : Mutable.Heap) (address : Nat) (k : Int), Mutable.Heap.Valid heap →
      Mutable.Value.HasType heap (.ref address) "list[list[int]]" →
      HeapCalls.Condition program (countEnv k) Solvers.P11Ast.requires_max_adjacent_product_0 heap →
      HeapCalls.Condition program (gridEnv address) Solvers.P11Ast.requires_max_adjacent_product_1 heap →
      (HeapCalls.call program "max_adjacent_product" [.ref address, .scalar (.int k)] heap).Supported) ∧
    (∀ (heap : Mutable.Heap) (address : Nat) (k : Int), Mutable.Heap.Valid heap →
      Mutable.Value.HasType heap (.ref address) "list[list[int]]" →
      HeapCalls.Condition program (countEnv k) Solvers.P11Ast.requires_max_adjacent_product_0 heap →
      HeapCalls.Condition program (gridEnv address) Solvers.P11Ast.requires_max_adjacent_product_1 heap →
      (HeapCalls.call P11Ast.module "naive" [.ref address, .scalar (.int k)] heap).Supported) := by
  constructor
  · intro heap address k _ typed count domain
    have nonnegative : 0 ≤ k := by simpa using (count_condition heap k).condition_iff.mp count
    lift k to Nat using nonnegative
    obtain ⟨grid, ⟨view⟩, size, rectangle⟩ := grid_domain_of_condition heap address typed domain
    exact (max_adjacent_product.exact heap address grid view 20 k (by omega) rectangle).returns.supported
  · intro heap address k valid typed _ _
    exact data.«project-euler-statements».data.py.project_euler_statements.p11.naive.«Supported» heap address k valid typed

end solvers.«11»
