import solvers.«96.py.lean.d».BoxLists

namespace solvers.«96».build_peers_units
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

def references (base count : Nat) : List Mutable.Value :=
  (List.range count).map fun index => .ref (base + 1 + index)

def buffers (units : List (Fin 9 → Cell)) : Mutable.Heap :=
  units.map fun cells => .list (peerLoop.encoded ((List.finRange 9).map cells))

def work (base : Mutable.Heap) (units : List (Fin 9 → Cell)) : Mutable.Heap :=
  HeapCollection.working base (references base.length units.length) (buffers units)

theorem root (base : Mutable.Heap) (units : List (Fin 9 → Cell)) :
    (work base units)[base.length]? = some (.list (references base.length units.length)) := HeapCollection.found _ _ _

theorem append_buffer (base : Mutable.Heap) (units : List (Fin 9 → Cell)) (cells : Fin 9 → Cell) :
    (work base units ++ [Mutable.Object.list (peerLoop.encoded ((List.finRange 9).map cells))]).set base.length
        (Mutable.Object.list (references base.length units.length ++ [Mutable.Value.ref (work base units).length])) = work base (units ++ [cells]) := by
  simp [work, buffers, references, HeapCollection.working, List.set_append_right, List.range_succ,
    List.append_assoc, Nat.add_assoc]
  omega

theorem append_collection (column : Bool) (index : Fin 9) (base : Mutable.Heap) (units : List (Fin 9 → Cell))
    (env : HeapCalls.Env) (globals : Mutable.Globals) (getUnits : env "units" = some (.ref base.length))
    (getFixed : env (fixed column) = some (number index.val)) :
    BlockContinues program env [.expr (.call (.attribute (.name "units") "append") [collection column])]
      (work base units) env (work base (units ++ [position column index])) globals := by
  have appended := Evaluates.append (Evaluates.lookup getUnits) (root base units)
    (collect column index env (work base units) globals getFixed)
    (HeapCollection.prefix_found (work base units) _ base.length _ (root base units))
  simpa only [append_buffer] using BlockContinues.discard appended

/-- Each complete outer row or column loop appends nine distinct unit lists. -/
theorem straight_units (column : Bool) (base : Mutable.Heap) (units : List (Fin 9 → Cell))
    (env : HeapCalls.Env) (globals : Mutable.Globals) (getUnits : env "units" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env
        [.forIn (.name (fixed column)) (.call (.name "range") [.constant (.int 9)])
          [.expr (.call (.attribute (.name "units") "append") [collection column])] []]
        (work base units) finalEnv (work base (units ++ (List.finRange 9).map (position column))) globals ∧
      finalEnv "units" = some (.ref base.length) := by
  let body : List Stmt := [.expr (.call (.attribute (.name "units") "append") [collection column])]
  let invariant := fun scope : HeapCalls.Env => scope "units" = some (.ref base.length)
  let update := fun current index => current ++ [position column index]
  obtain ⟨finalEnv, executed, kept⟩ := HeapFiniteRange.loop program (fixed column) body rfl 9 0 1 (by decide)
    (work base) update invariant globals
    (by intro scope index ready; cases column <;> simpa [fixed, invariant, setLocal] using ready)
    (by
      intro current scope index ready bound
      exact ⟨scope, append_collection column index base current scope globals ready (by simpa using bound), ready⟩)
    (List.finRange 9) 0 rfl (by omega) units env getUnits
  refine ⟨finalEnv, ?_, kept⟩
  apply BlockContinues.for_range (.cons Evaluates.constant .nil) rfl
  simpa [update, List.finRange_succ, List.finRange_zero, List.append_assoc] using executed

theorem append_box (r c : Fin 3) (base : Mutable.Heap) (units : List (Fin 9 → Cell))
    (env : HeapCalls.Env) (globals : Mutable.Globals) (getUnits : env "units" = some (.ref base.length))
    (getRow : env "br" = some (number r.val)) (getColumn : env "bc" = some (number c.val)) :
    ∃ finalEnv, BlockContinues program env boxBody (work base units) finalEnv (work base (units ++ [boxCell r c])) globals ∧
      finalEnv "units" = some (.ref base.length) ∧ finalEnv "br" = env "br" := by
  obtain ⟨finalEnv, collected, getBox, preserved⟩ := collect_box r c env (work base units) globals getRow getColumn
  have unitsLookup := (preserved "units" (by decide)).trans getUnits
  have found := HeapCollection.prefix_found (work base units)
    [.list (peerLoop.encoded ((List.finRange 9).map (boxCell r c)))] base.length _ (root base units)
  have appended := Evaluates.append (module := program) (globals := globals) (Evaluates.lookup unitsLookup)
    found (Evaluates.lookup getBox) found
  refine ⟨finalEnv, ?_, unitsLookup, preserved "br" (by decide)⟩
  have finished := BlockContinues.discard appended
  rw [append_buffer] at finished
  exact collected.append finished

abbrev boxesInRow : Stmt := .forIn (.name "bc") (.call (.name "range") [.constant (.int 3)]) boxBody []

theorem box_row_units (r : Fin 3) (base : Mutable.Heap) (units : List (Fin 9 → Cell))
    (env : HeapCalls.Env) (globals : Mutable.Globals) (getUnits : env "units" = some (.ref base.length))
    (getRow : env "br" = some (number r.val)) :
    ∃ finalEnv, BlockContinues program env [boxesInRow] (work base units) finalEnv
        (work base (units ++ (List.finRange 3).map (boxCell r))) globals ∧
      finalEnv "units" = some (.ref base.length) := by
  let invariant := fun scope : HeapCalls.Env => scope "units" = some (.ref base.length) ∧ scope "br" = some (number r.val)
  let update := fun current c => current ++ [boxCell r c]
  obtain ⟨finalEnv, executed, kept⟩ := HeapFiniteRange.loop program "bc" boxBody rfl 3 0 1 (by decide)
    (work base) update invariant globals
    (by
      intro scope c ready
      exact ⟨by simpa [setLocal] using ready.1, by simpa [setLocal] using ready.2⟩)
    (by
      intro current scope c ready bound
      obtain ⟨afterEnv, executed, unitsKept, rowKept⟩ :=
        append_box r c base current scope globals ready.1 ready.2 (by simpa using bound)
      exact ⟨afterEnv, executed, unitsKept, rowKept.trans ready.2⟩)
    (List.finRange 3) 0 rfl (by omega) units env ⟨getUnits, getRow⟩
  refine ⟨finalEnv, ?_, kept.1⟩
  apply BlockContinues.for_range (.cons Evaluates.constant .nil) rfl
  simpa [update, List.finRange_succ, List.finRange_zero, List.append_assoc] using executed

theorem box_units (base : Mutable.Heap) (units : List (Fin 9 → Cell))
    (env : HeapCalls.Env) (globals : Mutable.Globals) (getUnits : env "units" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env
        [.forIn (.name "br") (.call (.name "range") [.constant (.int 3)]) [boxesInRow] []]
        (work base units) finalEnv
        (work base (units ++ (List.finRange 3).flatMap fun r => (List.finRange 3).map (boxCell r))) globals ∧
      finalEnv "units" = some (.ref base.length) := by
  let invariant := fun scope : HeapCalls.Env => scope "units" = some (.ref base.length)
  let update := fun current r => current ++ (List.finRange 3).map (boxCell r)
  obtain ⟨finalEnv, executed, kept⟩ := HeapFiniteRange.loop program "br" [boxesInRow] rfl 3 0 1 (by decide)
    (work base) update invariant globals
    (by intro scope r ready; simpa [invariant, setLocal] using ready)
    (by
      intro current scope r ready bound
      exact box_row_units r base current scope globals ready (by simpa using bound))
    (List.finRange 3) 0 rfl (by omega) units env getUnits
  refine ⟨finalEnv, ?_, kept⟩
  apply BlockContinues.for_range (.cons Evaluates.constant .nil) rfl
  simpa [update, List.finRange_succ, List.finRange_zero, List.append_assoc] using executed

/-- The first four source statements construct all 27 units with fresh, distinct backing lists. -/
theorem units (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals) :
    ∃ finalEnv, BlockContinues program env (definition.2.take 4) heap finalEnv
        (work heap ((List.finRange 27).map Units.cells)) globals ∧
      finalEnv "units" = some (.ref heap.length) := by
  let ready := setLocal env "units" (.ref heap.length)
  have initialized : BlockContinues program env [.assign [.name "units"] (.list [])] heap ready (work heap []) globals :=
    BlockContinues.assign (Evaluates.list .nil)
  obtain ⟨afterRows, rows, rowsLookup⟩ := straight_units false heap [] ready globals (by simp [ready, setLocal])
  obtain ⟨afterColumns, columns, columnsLookup⟩ := straight_units true heap ((List.finRange 9).map (position false))
    afterRows globals rowsLookup
  obtain ⟨finalEnv, boxes, finalLookup⟩ := box_units heap
    ((List.finRange 9).map (position false) ++ (List.finRange 9).map (position true)) afterColumns globals columnsLookup
  have allUnits : ((List.finRange 9).map (position false) ++ (List.finRange 9).map (position true)) ++
      (List.finRange 3).flatMap (fun r => (List.finRange 3).map (boxCell r)) = (List.finRange 27).map Units.cells := rfl
  refine ⟨finalEnv, ?_, finalLookup⟩
  rw [allUnits] at boxes
  exact initialized.append (rows.append (columns.append boxes))

/-- The allocated source layout is precisely the complete unit representation used by propagation and search. -/
theorem table_view (heap : Mutable.Heap) : (table heap.length).View (work heap ((List.finRange 27).map Units.cells)) := by
  constructor
  · have found := root heap ((List.finRange 27).map Units.cells)
    have same : references heap.length 27 = UnitTable.references (table heap.length).units := rfl
    simpa only [List.length_map, List.length_finRange, same] using found
  · intro unit member
    simp only [table, List.mem_ofFn] at member
    obtain ⟨index, rfl⟩ := member
    have offset : heap.length + 1 + index.val - heap.length = index.val + 1 := by omega
    simp [work, HeapCollection.working, buffers,
      List.getElem?_append_right (show heap.length ≤ heap.length + 1 + index.val by omega), offset, index.isLt]
    intro cell
    rfl

end
end solvers.«96».build_peers_units
