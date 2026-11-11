import solvers.«84.py.lean.d».TransitionPositionLoop

namespace solvers.«84».transitions
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

def code : List Stmt := (executableBody outcomes.definition.2).drop 4

theorem source : code = [.assign [.name "transitions"] (.list []), positionLoop, .return (.name "transitions")] := rfl

def View (sides : Nat) (heap : Mutable.Heap) (value : Mutable.Value) : Prop :=
  ∃ root rows, value = .ref root ∧ heap[root]? = some (.list (references rows)) ∧ Views sides heap allStates rows

def Post (sides : Nat) (base : Mutable.Heap) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ extra, after = base ++ extra ∧ View sides after value

/-- The actual nested source loops return all transition rows and preserve every caller object. -/
theorem collect (sides : Nat) (positive : 0 < sides) (base : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (source tableRoot : Nat)
    (sourceView : outcomes.View base source (outcomes.all sides)) (table : transition_row.Table base tableRoot)
    (getSource : env "outcomes" = some (.ref source)) (getTable : env "landings" = some (.ref tableRoot)) :
    ∃ (rows : List Row) (garbage : Mutable.Heap),
      BlockReturns program env code base (.ref base.length) (working base rows garbage) globals ∧
      Views sides (working base rows garbage) allStates rows := by
  let start := setLocal env "transitions" (.ref base.length)
  have ready : Ready start source tableRoot base.length :=
    ⟨by simpa [start, setLocal] using getSource, by simpa [start, setLocal] using getTable, by simp [start, setLocal]⟩
  obtain ⟨rows, garbage, finalEnv, ran, views, finalReady⟩ :=
    position_result sides positive base [] [] [] start globals source tableRoot sourceView table .nil ready
  refine ⟨rows, garbage, ?_, by simpa using views⟩
  rw [transitions.source]
  exact (BlockContinues.assign (Evaluates.list .nil)).returns
    (ran.returns (BlockReturns.return_value (Evaluates.lookup finalReady.getCollector)))

theorem collect_result (sides : Nat) (positive : 0 < sides) (base : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (source tableRoot : Nat)
    (sourceView : outcomes.View base source (outcomes.all sides)) (table : transition_row.Table base tableRoot)
    (getSource : env "outcomes" = some (.ref source)) (getTable : env "landings" = some (.ref tableRoot)) :
    HeapMemoCalls.BlockReturnsRel program resolve_landing.pool.spec env code base (Post sides base) globals := by
  obtain ⟨rows, garbage, ran, views⟩ := collect sides positive base env globals source tableRoot sourceView table getSource getTable
  intro anchors stack
  have lifted := HeapMemoCalls.Runs.from_heap (HeapCalls.Extends.refl program) resolve_landing.pool.disjoint
    (ran anchors stack) (TuplePool.preserves_append resolve_landing.pool.catalog base _)
  apply lifted.consequence
  intro final same
  subst final
  exact ⟨.ref base.length, working base rows garbage, rfl, .list (references rows) :: garbage,
    rfl, base.length, rows, rfl, HeapCollection.found _ _ _, views⟩

end solvers.«84».transitions
