import solvers.«84.py.lean.d».TransitionTableView

namespace solvers.«84».transitions
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary
set_option Elab.async false

/-- Construct one private row, then append its actual reference to the transition list. -/
theorem collect_row (sides : Nat) (positive : 0 < sides) (base garbage : Mutable.Heap)
    (states : List (Nat × Nat)) (rows : List Row) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (source tableRoot pos doubles : Nat)
    (sourceView : outcomes.View base source (outcomes.all sides)) (table : transition_row.Table base tableRoot)
    (views : Views sides (working base rows garbage) states rows) (ready : Ready env source tableRoot base.length)
    (getPos : env "pos" = some (number pos)) (getDoubles : env "doubles" = some (number doubles)) :
    ∃ (row : Row) (extra : Mutable.Heap) (finalEnv : HeapCalls.Env),
      BlockContinues program env rowBody (working base rows garbage) finalEnv
        (working base (rows ++ [row]) (garbage ++ extra)) globals ∧
      Views sides (working base (rows ++ [row]) (garbage ++ extra)) (states ++ [(pos, doubles)]) (rows ++ [row]) ∧
      Ready finalEnv source tableRoot base.length ∧ finalEnv "pos" = some (number pos) := by
  obtain ⟨entries, extra, finalEnv, built, rowFound, getRow, frame, counts⟩ :=
    transition_row.model_result env (working base rows garbage) globals sides source tableRoot pos doubles positive
      (sourceView.append _) (table.append _) ready.getSource ready.getTable getPos getDoubles
  let row : Row := ⟨(working base rows garbage).length, entries⟩
  have nextReady : Ready finalEnv source tableRoot base.length :=
    ⟨(frame _ (by decide) (by decide)).trans ready.getSource,
      (frame _ (by decide) (by decide)).trans ready.getTable,
      (frame _ (by decide) (by decide)).trans ready.getCollector⟩
  have collector := HeapCollection.prefix_found (working base rows garbage) extra base.length _ (HeapCollection.found _ _ _)
  have appended := BlockContinues.discard (module := program) (globals := globals) (Evaluates.append
    (Evaluates.lookup nextReady.getCollector) collector (Evaluates.lookup getRow) collector)
  have allViews := (views.append extra).snoc (pos, doubles) row ⟨rowFound, counts⟩
  have stored := allViews.store base.length (references rows) (references rows ++ [.ref row.address]) collector
  have layout : ((working base rows garbage) ++ extra).set base.length (.list (references rows ++ [.ref row.address])) =
      working base (rows ++ [row]) (garbage ++ extra) := by
    rw [HeapCollection.append, HeapCollection.store]
    simp [working, references]
  rw [layout] at stored
  refine ⟨row, extra, finalEnv, ?_, stored, nextReady, (frame _ (by decide) (by decide)).trans getPos⟩
  rw [row_source]
  have ran := built.append appended
  simpa only [row, layout] using ran

end solvers.«84».transitions
