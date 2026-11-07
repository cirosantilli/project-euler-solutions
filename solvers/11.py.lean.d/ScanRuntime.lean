import solvers.«11.py.lean.d».DirectionRuntime
import lang.py.FormalSW.HeapRuntimeCostUnpacking

namespace solvers.«11».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime ProjectEulerStatements.Grid

theorem directions.loop (heap : Mutable.Heap) (address table : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid) (choices : List Direction)
    (tableView : DirectionView heap table choices) (columns r c n position : Nat)
    (size : grid.length = 20) (width : columns = 20)
    (rectangle : ∀ row ∈ grid, row.length = columns) (insideRow : r < grid.length) (insideColumn : c < columns)
    (env : HeapCalls.Env) (best : Int)
    (getGrid : env "grid" = some (.ref address))
    (getRow : env "i" = some (.scalar (.int r)))
    (getColumn : env "j" = some (.scalar (.int c)))
    (getCount : env "k" = some (.scalar (.int n)))
    (getRows : env "n" = some (.scalar (.int grid.length)))
    (getColumns : env "m" = some (.scalar (.int columns)))
    (getBest : env "best" = some (.scalar (.int best))) :
    ∃ finalEnv, ∃ answer : Int, BlockContinuesIn .word program env
      [.unpackStepLoop ["di", "dj"] (.ref table) position directionBody []] heap (1111*(choices.length-position)+5) finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int answer)) ∧
      (∀ name, name ∉ ["di", "dj", "ni", "nj", "t", "prod", "best"] → finalEnv name = env name) := by
  by_cases active : position < choices.length
  · have pairIndex : position < tableView.pairs.length := by rw [tableView.size]; exact active
    let direction := choices[position]
    let pair := tableView.pairs[position]
    let fields : List Mutable.Value := [.scalar (.int direction.rowStep), .scalar (.int direction.columnStep)]
    let boundEnv := HeapCalls.assignNames env ["di", "dj"] fields
    have preserved (name : String) (notDi : name ≠ "di") (notDj : name ≠ "dj") : boundEnv name = env name := by
      simp [boundEnv, fields, HeapCalls.assignNames, HeapCalls.setLocal, notDi, notDj]
    obtain ⟨middle, nextBest, step, result, saved⟩ := direction_cost direction heap address grid view columns r c n
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
    have boundedStep : BlockContinuesIn .word program boundEnv directionBody heap 1100 middle heap :=
      step.mono (by rw [size, width]; decide)
    obtain ⟨finalEnv, answer, rest, finalBest, finalSaved⟩ := directions.loop heap address table grid view choices tableView
      columns r c n (position + 1) size width rectangle insideRow insideColumn middle nextBest
      (by rw [saved "grid" (by simp), preserved "grid" (by decide) (by decide)]; exact getGrid)
      (by rw [saved "i" (by simp), preserved "i" (by decide) (by decide)]; exact getRow)
      (by rw [saved "j" (by simp), preserved "j" (by decide) (by decide)]; exact getColumn)
      (by rw [saved "k" (by simp), preserved "k" (by decide) (by decide)]; exact getCount)
      (by rw [saved "n" (by simp), preserved "n" (by decide) (by decide)]; exact getRows)
      (by rw [saved "m" (by simp), preserved "m" (by decide) (by decide)]; exact getColumns) result
    refine ⟨finalEnv, answer, ?_, finalBest, ?_⟩
    · apply BlockContinuesIn.mono (small := (1100+(1111*(choices.length-(position+1))+5))+2+2+7)
        (larger := by omega)
      apply BlockContinuesIn.unpack_step (items := tableView.pairs.map Mutable.Value.ref)
        (pair := pair) (values := fields) tableView.found
      · simp [List.getElem?_eq_getElem (show position < (tableView.pairs.map Mutable.Value.ref).length by simpa using pairIndex), pair]
      · exact tableView.contents position pairIndex
      · rfl
      · simpa only [directionBody, columnsBody, rowsBody, sourceBody, Calls.loopContinuation] using boundedStep.append rest
    · intro name untouched
      rw [finalSaved name untouched]
      have notDi : name ≠ "di" := by intro h; exact untouched (by simp [h])
      have notDj : name ≠ "dj" := by intro h; exact untouched (by simp [h])
      have notOther : name ∉ ["ni", "nj", "t", "prod", "best"] := by
        intro member
        exact untouched (by simp only [List.mem_cons] at member ⊢; tauto)
      rw [saved name notOther, preserved name notDi notDj]
  · have outside : choices.length ≤ position := by omega
    refine ⟨env, best, ?_, getBest, by intros; rfl⟩
    simpa [Nat.sub_eq_zero_of_le outside] using
      (BlockContinuesIn.unpack_done (names := ["di", "dj"]) (body := directionBody) tableView.found
        (List.getElem?_eq_none (by simpa [tableView.size] using outside)) BlockContinuesIn.nil)
termination_by choices.length - position
decreasing_by omega

theorem directions.result (heap : Mutable.Heap) (address table : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid)
    (tableView : DirectionView heap table directions) (columns r c n : Nat)
    (size : grid.length = 20) (width : columns = 20)
    (rectangle : ∀ row ∈ grid, row.length = columns) (insideRow : r < grid.length) (insideColumn : c < columns)
    (env : HeapCalls.Env) (best : Int)
    (getGrid : env "grid" = some (.ref address))
    (getRow : env "i" = some (.scalar (.int r)))
    (getColumn : env "j" = some (.scalar (.int c)))
    (getCount : env "k" = some (.scalar (.int n)))
    (getRows : env "n" = some (.scalar (.int grid.length)))
    (getColumns : env "m" = some (.scalar (.int columns)))
    (getBest : env "best" = some (.scalar (.int best)))
    (getTable : env "directions" = some (.ref table)) :
    ∃ finalEnv, ∃ answer : Int, BlockContinuesIn .word program env columnsBody heap (4454) finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int answer)) ∧
      (∀ name, name ∉ ["di", "dj", "ni", "nj", "t", "prod", "best"] → finalEnv name = env name) := by
  obtain ⟨finalEnv, answer, loop, result, saved⟩ := directions.loop heap address table grid view directions tableView
    columns r c n 0 size width rectangle insideRow insideColumn env best getGrid getRow getColumn getCount getRows getColumns getBest
  refine ⟨finalEnv, answer, ?_, result, saved⟩
  exact BlockContinuesIn.for_unpack_name rfl getTable tableView.found loop


theorem column_loop (heap : Mutable.Heap) (address table : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid)
    (tableView : DirectionView heap table directions) (columns r n position : Nat)
    (size : grid.length = 20) (width : columns = 20)
    (rectangle : ∀ row ∈ grid, row.length = columns) (insideRow : r < grid.length)
    (env : HeapCalls.Env) (best : Int)
    (getGrid : env "grid" = some (.ref address))
    (getRow : env "i" = some (.scalar (.int r)))
    (getCount : env "k" = some (.scalar (.int n)))
    (getRows : env "n" = some (.scalar (.int grid.length)))
    (getColumns : env "m" = some (.scalar (.int columns)))
    (getBest : env "best" = some (.scalar (.int best)))
    (getTable : env "directions" = some (.ref table)) :
    ∃ finalEnv, ∃ answer : Int, BlockContinuesIn .word program env
      [.rangeStepLoop "j" position columns 1 columnsBody []] heap (4458*(columns-position)+4) finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int answer)) ∧
      (∀ name, name ∉ ["j", "di", "dj", "ni", "nj", "t", "prod", "best"] → finalEnv name = env name) := by
  by_cases active : position < columns
  · let boundEnv := HeapCalls.setLocal env "j" (.scalar (.int position))
    obtain ⟨middle, nextBest, step, result, saved⟩ := directions.result heap address table grid view tableView
      columns r position n size width rectangle insideRow active boundEnv best
      (by simpa [boundEnv, HeapCalls.setLocal] using getGrid)
      (by simpa [boundEnv, HeapCalls.setLocal] using getRow) rfl
      (by simpa [boundEnv, HeapCalls.setLocal] using getCount)
      (by simpa [boundEnv, HeapCalls.setLocal] using getRows)
      (by simpa [boundEnv, HeapCalls.setLocal] using getColumns)
      (by simpa [boundEnv, HeapCalls.setLocal] using getBest)
      (by simpa [boundEnv, HeapCalls.setLocal] using getTable)
    obtain ⟨finalEnv, answer, rest, finalBest, finalSaved⟩ := column_loop heap address table grid view tableView
      columns r n (position + 1) size width rectangle insideRow middle nextBest
      (by rw [saved "grid" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getGrid)
      (by rw [saved "i" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getRow)
      (by rw [saved "k" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getCount)
      (by rw [saved "n" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getRows)
      (by rw [saved "m" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getColumns) result
      (by rw [saved "directions" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getTable)
    refine ⟨finalEnv, answer, ?_, finalBest, ?_⟩
    · have run := BlockContinuesIn.range_step (env := env) (name := "j")
          (by simp [Calls.rangeContinues, active] : Calls.rangeContinues (position : Int) columns 1 = true)
          (by simpa [columnsBody, Calls.loopContinuation, Calls.loopControl, Nat.cast_add, Nat.cast_one] using step.append rest)
      exact run.mono (by omega)
    · intro name untouched
      rw [finalSaved name untouched, saved name (by intro member; exact untouched (List.mem_cons_of_mem _ member))]
      have notColumn : name ≠ "j" := by intro equal; exact untouched (by simp [equal])
      simp [boundEnv, HeapCalls.setLocal, notColumn]
  · refine ⟨env, best, ?_, getBest, by intros; rfl⟩
    simpa [Nat.sub_eq_zero_of_le (show columns ≤ position by omega)] using
      (BlockContinuesIn.range_done (name := "j") (env := env) (body := columnsBody)
        (by simp [Calls.rangeContinues, active] : Calls.rangeContinues (position : Int) columns 1 = false) BlockContinuesIn.nil)
termination_by columns - position
decreasing_by omega

theorem rows.loop (heap : Mutable.Heap) (address table : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid)
    (tableView : DirectionView heap table directions) (columns n position : Nat)
    (size : grid.length = 20) (width : columns = 20)
    (rectangle : ∀ row ∈ grid, row.length = columns) (env : HeapCalls.Env) (best : Int)
    (getGrid : env "grid" = some (.ref address))
    (getCount : env "k" = some (.scalar (.int n)))
    (getRows : env "n" = some (.scalar (.int grid.length)))
    (getColumns : env "m" = some (.scalar (.int columns)))
    (getBest : env "best" = some (.scalar (.int best)))
    (getTable : env "directions" = some (.ref table)) :
    ∃ finalEnv, ∃ answer : Int, BlockContinuesIn .word program env
      [.rangeStepLoop "i" position grid.length 1 rowsBody []] heap (89173*(grid.length-position)+4) finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int answer)) ∧
      (∀ name, name ∉ ["i", "j", "di", "dj", "ni", "nj", "t", "prod", "best"] → finalEnv name = env name) := by
  by_cases active : position < grid.length
  · let boundEnv := HeapCalls.setLocal env "i" (.scalar (.int position))
    obtain ⟨middle, nextBest, columnsRan, result, saved⟩ := column_loop heap address table grid view tableView
      columns position n 0 size width rectangle active boundEnv best
      (by simpa [boundEnv, HeapCalls.setLocal] using getGrid) rfl
      (by simpa [boundEnv, HeapCalls.setLocal] using getCount)
      (by simpa [boundEnv, HeapCalls.setLocal] using getRows)
      (by simpa [boundEnv, HeapCalls.setLocal] using getColumns)
      (by simpa [boundEnv, HeapCalls.setLocal] using getBest)
      (by simpa [boundEnv, HeapCalls.setLocal] using getTable)
    have step : BlockContinuesIn .word program boundEnv rowsBody heap 89169 middle heap := by
      have run := BlockContinuesIn.for_range_bound
        (EvaluatesIn.lookup (name := "m") (by simpa [boundEnv, HeapCalls.setLocal] using getColumns)) columnsRan
      exact run.mono (by rw [width])
    obtain ⟨finalEnv, answer, rest, finalBest, finalSaved⟩ := rows.loop heap address table grid view tableView
      columns n (position + 1) size width rectangle middle nextBest
      (by rw [saved "grid" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getGrid)
      (by rw [saved "k" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getCount)
      (by rw [saved "n" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getRows)
      (by rw [saved "m" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getColumns) result
      (by rw [saved "directions" (by simp)]; simpa [boundEnv, HeapCalls.setLocal] using getTable)
    refine ⟨finalEnv, answer, ?_, finalBest, ?_⟩
    · have run := BlockContinuesIn.range_step (env := env) (name := "i")
          (by simp [Calls.rangeContinues, active] : Calls.rangeContinues (position : Int) grid.length 1 = true)
          (by simpa [rowsBody, columnsBody, Calls.loopContinuation, Calls.loopControl, Nat.cast_add, Nat.cast_one] using step.append rest)
      exact run.mono (by omega)
    · intro name untouched
      rw [finalSaved name untouched, saved name (by intro member; exact untouched (List.mem_cons_of_mem _ member))]
      have notRow : name ≠ "i" := by intro equal; exact untouched (by simp [equal])
      simp [boundEnv, HeapCalls.setLocal, notRow]
  · refine ⟨env, best, ?_, getBest, by intros; rfl⟩
    simpa [Nat.sub_eq_zero_of_le (show grid.length ≤ position by omega)] using
      (BlockContinuesIn.range_done (name := "i") (env := env) (body := rowsBody)
        (by simp [Calls.rangeContinues, active] : Calls.rangeContinues (position : Int) grid.length 1 = false) BlockContinuesIn.nil)
termination_by grid.length - position
decreasing_by omega


end solvers.«11».runtimeCost
