import solvers.«79.py.lean.d».Execution

namespace solvers.«79».source
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
open solvers.«79».graph
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

theorem processedOutcome_prefix (data : GraphData) (attempts rest : List String)
    (valid : ∀ s ∈ attempts, ValidAttempt s) :
    processedOutcome data (attempts ++ rest) =
      processedOutcome (attempts.foldl GraphData.extend data) rest := by
  induction attempts generalizing data with
  | nil => rfl
  | cons s tail ih =>
      have head := valid s (by simp)
      simpa only [List.cons_append, List.foldl_cons, processedOutcome, GraphData.process, if_pos head]
        using ih (data.extend s) (fun value member => valid value (List.mem_cons_of_mem _ member))

theorem ConsumersAgree.traversal (state next : Storage) (checked attempts : List String)
    (agree : ConsumersAgree state checked) (valid : ∀ s ∈ attempts, ValidAttempt s)
    (data : next.graphData = attempts.foldl GraphData.extend state.graphData) :
    ConsumersAgree next (checked ++ attempts) := by
  intro rest
  have matched := agree (attempts ++ rest)
  rw [processedOutcome_prefix _ _ _ valid, p79.validatedOutcome_prefix _ _ _ valid] at matched
  simpa only [data] using matched

end solvers.«79».source
