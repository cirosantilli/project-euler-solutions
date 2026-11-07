import solvers.«11.py.lean.d».ProductRuntime

namespace solvers.«11».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime ProjectEulerStatements.Grid

private def endpointRow : Expr := match directionBody with
  | .assign _ expression :: _ => expression | _ => .constant .none
private def endpointColumn : Expr := match directionBody with
  | _ :: .assign _ expression :: _ => expression | _ => .constant .none
private def endpointCondition : Expr := match directionBody with
  | [_, _, .ifThen expression _ _] => expression | _ => .constant .none

private theorem endpointCondition.result (heap : Mutable.Heap) (env : HeapCalls.Env) (ni nj : Int) (rows columns : Nat)
    (getNI : env "ni" = some (.scalar (.int ni))) (getNJ : env "nj" = some (.scalar (.int nj)))
    (getRows : env "n" = some (.scalar (.int rows))) (getColumns : env "m" = some (.scalar (.int columns))) :
    EvaluatesIn .word program env endpointCondition heap 30
      (.scalar (.bool (decide ((0 ≤ ni ∧ ni < rows) ∧ (0 ≤ nj ∧ nj < columns))))) heap := by
  have row := EvaluatesIn.nonnegative_below (module := program) (heap := heap)
    (EvaluatesIn.lookup getNI) (EvaluatesIn.lookup getRows)
  have column := EvaluatesIn.nonnegative_below (module := program) (heap := heap)
    (EvaluatesIn.lookup getNJ) (EvaluatesIn.lookup getColumns)
  by_cases validRow : 0 ≤ ni ∧ ni < rows
  · simp only [validRow, true_and]
    exact EvaluatesIn.boolean_next (flag := true) (by simpa [validRow] using row) rfl rfl
      (EvaluatesIn.boolean_last column rfl)
  · simp only [validRow, false_and, decide_false]
    exact (EvaluatesIn.boolean_stop (flag := false) (by simpa [validRow] using row) rfl rfl).mono (by omega)

theorem direction_cost (direction : Direction) (heap : Mutable.Heap) (address : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid) (columns r c n : Nat)
    (rectangle : ∀ row ∈ grid, row.length = columns) (insideRow : r < grid.length) (insideColumn : c < columns)
    (env : HeapCalls.Env) (best : Int)
    (getGrid : env "grid" = some (.ref address))
    (getRow : env "i" = some (.scalar (.int r)))
    (getColumn : env "j" = some (.scalar (.int c)))
    (getDi : env "di" = some (.scalar (.int direction.rowStep)))
    (getDj : env "dj" = some (.scalar (.int direction.columnStep)))
    (getCount : env "k" = some (.scalar (.int n)))
    (getRows : env "n" = some (.scalar (.int grid.length)))
    (getColumns : env "m" = some (.scalar (.int columns)))
    (getBest : env "best" = some (.scalar (.int best))) :
    ∃ finalEnv, ∃ answer : Int, BlockContinuesIn .word program env directionBody heap
      (49 * max grid.length columns + 100) finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int answer)) ∧
      (∀ name, name ∉ ["ni", "nj", "t", "prod", "best"] → finalEnv name = env name) := by
  let ni : Int := r + (n - 1 : Int) * direction.rowStep
  let nj : Int := c + (n - 1 : Int) * direction.columnStep
  let rowEnv := HeapCalls.setLocal env "ni" (.scalar (.int ni))
  let endEnv := HeapCalls.setLocal rowEnv "nj" (.scalar (.int nj))
  have rowAssigned : BlockContinuesIn .word program env [.assign [.name "ni"] endpointRow] heap 18 rowEnv heap :=
    BlockContinuesIn.assign (EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup getRow)
      (EvaluatesIn.binary (work := 1)
        (EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup getCount) EvaluatesIn.constant rfl rfl)
        (EvaluatesIn.lookup getDi) rfl rfl) rfl rfl)
  have columnAssigned : BlockContinuesIn .word program rowEnv [.assign [.name "nj"] endpointColumn] heap 18 endEnv heap :=
    BlockContinuesIn.assign (EvaluatesIn.binary (work := 1)
      (EvaluatesIn.lookup (by simpa [rowEnv, HeapCalls.setLocal] using getColumn))
      (EvaluatesIn.binary (work := 1)
        (EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup (by simpa [rowEnv, HeapCalls.setLocal] using getCount))
          EvaluatesIn.constant rfl rfl)
        (EvaluatesIn.lookup (by simpa [rowEnv, HeapCalls.setLocal] using getDj)) rfl rfl) rfl rfl)
  have tested : EvaluatesIn .word program endEnv endpointCondition heap 30
      (.scalar (.bool (decide (ValidStart direction grid.length columns n r c)))) heap := by
    have test := endpointCondition.result heap endEnv ni nj grid.length columns
      (by simp [endEnv, rowEnv, HeapCalls.setLocal]) rfl
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getRows)
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getColumns)
    simpa only [ValidStart, insideRow, insideColumn, true_and] using test
  by_cases valid : ValidStart direction grid.length columns n r c
  · obtain ⟨finalEnv, answer, product, result, preserved⟩ := candidate_cost direction heap address grid view columns r c n
      rectangle valid endEnv best
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getGrid)
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getRow)
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getColumn)
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getDi)
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getDj)
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getCount)
      (by simpa [endEnv, rowEnv, HeapCalls.setLocal] using getBest)
    have branch := BlockContinuesIn.branch (yes := candidateBody) (no := []) (flag := true)
      (by simpa [valid] using tested) rfl product
    have boundedCount := valid.count_le
    refine ⟨finalEnv, answer, (rowAssigned.append (columnAssigned.append branch)).mono (by omega), result, ?_⟩
    intro name untouched
    simp only [List.mem_cons, not_or] at untouched
    rw [preserved name untouched.2.2.1 untouched.2.2.2.1 untouched.2.2.2.2.1]
    simp [endEnv, rowEnv, HeapCalls.setLocal, untouched.1, untouched.2.1]
  · have branch := BlockContinuesIn.branch (yes := candidateBody) (no := []) (flag := false)
      (by simpa [valid] using tested) rfl BlockContinuesIn.nil
    refine ⟨endEnv, best, (rowAssigned.append (columnAssigned.append branch)).mono (by omega), ?_, ?_⟩
    · simpa [valid, endEnv, rowEnv, HeapCalls.setLocal] using getBest
    · intro name untouched
      simp only [List.mem_cons, not_or] at untouched
      simp [endEnv, rowEnv, HeapCalls.setLocal, untouched.1, untouched.2.1]


end solvers.«11».runtimeCost
