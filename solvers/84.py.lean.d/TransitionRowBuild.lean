import solvers.«84.py.lean.d».TransitionRollLoop

open FormalSW.PyAst ProjectEulerSolutions

namespace solvers.«84».outcomes

theorem View.append {before : Mutable.Heap} {source : Nat} {values : List Outcome}
    (view : View before source values) (extra : Mutable.Heap) : View (before ++ extra) source values := by
  obtain ⟨items, present, represented⟩ := view
  refine ⟨items, HeapCollection.prefix_found _ _ _ _ present, ?_⟩
  clear present
  induction represented with
  | nil => exact .nil
  | cons head tail ih =>
      obtain ⟨address, rfl, tupleFound⟩ := head
      exact .cons ⟨address, rfl, HeapCollection.prefix_found _ _ _ _ tupleFound⟩ ih

end solvers.«84».outcomes

namespace solvers.«84».transition_row
open HeapCalls NaturalDictionary
set_option Elab.async false

theorem Table.append {before : Mutable.Heap} {source : Nat} (table : Table before source) (extra : Mutable.Heap) :
    Table (before ++ extra) source := by
  obtain ⟨values, present, length, contents⟩ := table
  refine ⟨values, HeapCollection.prefix_found _ _ _ _ present, length, ?_⟩
  intro square bound
  obtain ⟨dictionary, selected, view⟩ := contents square bound
  exact ⟨dictionary, selected, view.append extra⟩

def initialCode : List Stmt := (loopBody (loopBody (executableBody outcomes.definition.2) 5) 0).take 2

theorem initial_source : initialCode = [.assign [.name "row"] (.dict []), rolls.loop] := rfl

def LocalsFrame (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ≠ "row" → name ∉ rolls.rebound → after name = before name

/-- Allocate and construct a complete row, preserving the entire heap that preceded its allocation. -/
theorem build_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (source tableRoot pos doubles : Nat) (original : List outcomes.Outcome)
    (sourceView : outcomes.View heap source original) (table : Table heap tableRoot)
    (getSource : env "outcomes" = some (.ref source)) (getTable : env "landings" = some (.ref tableRoot))
    (getPos : env "pos" = some (number pos)) (getDoubles : env "doubles" = some (number doubles))
    (nonnegative : ∀ outcome ∈ original, 0 ≤ outcome.move ∧ 0 ≤ outcome.count) :
    ∃ (result : Entries) (extra : Mutable.Heap) (finalEnv : HeapCalls.Env),
      BlockContinues program env initialCode heap finalEnv (heap ++ extra) globals ∧ Found (heap ++ extra) heap.length result ∧
      finalEnv "row" = some (.ref heap.length) ∧ LocalsFrame env finalEnv ∧
      ∀ target, ((lookup result target).getD 0 : Int) = rolls.total pos doubles original target := by
  let start := heap ++ [Mutable.Object.dict []]
  let localEnv := setLocal env "row" (.ref heap.length)
  obtain ⟨result, finalHeap, finalEnv, ran, finalFound, frame, localFrame, counts⟩ :=
    rolls.result localEnv start globals source tableRoot heap.length pos doubles original []
      (sourceView.append _) (table.append _) (by simp [Found, start])
      (by simpa [localEnv, setLocal] using getSource) (by simpa [localEnv, setLocal] using getTable)
      (by simp [localEnv, setLocal]) (by simpa [localEnv, setLocal] using getPos)
      (by simpa [localEnv, setLocal] using getDoubles) nonnegative
  have kept : heap <+: finalHeap := by
    apply List.prefix_iff_getElem?.mpr
    intro index inside
    exact frame index heap[index] (by omega) (HeapCollection.prefix_found heap _ index _ (List.getElem?_eq_getElem inside))
  obtain ⟨extra, same⟩ := kept
  rw [← same] at ran finalFound
  refine ⟨result, extra, finalEnv, ?_, finalFound, ?_, ?_, ?_⟩
  · rw [initial_source]
    exact (BlockContinues.assign (Evaluates.dictionary .nil)).append ran
  · exact (localFrame "row" (by decide)).trans (by simp [localEnv, setLocal])
  · intro name different untouched
    rw [localFrame name untouched]
    simp [localEnv, setLocal, different]
  · intro target
    simpa [lookup] using counts target

theorem model_total (sides pos doubles target : Nat) (positive : 0 < sides) :
    rolls.total pos doubles (outcomes.all sides) target =
      ((Monopoly.turnWeights sides (pos, doubles)).map fun entry =>
        if target = Monopoly.encode entry.1.1 entry.1.2 then (entry.2 : Int) else 0).sum := by
  rw [rolls.total, outcomes.all_sum_list sides positive (fun move flag => rowWeight pos doubles move flag target)]
  simp only [rowWeight, outcomes.category_roll, Monopoly.turnWeights, List.flatMap_def,
    List.map_flatten, List.sum_flatten, List.map_map, Function.comp_def]

/-- With the generated dice table, each complete source row equals ordered dice/card enumeration. -/
theorem model_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (sides source tableRoot pos doubles : Nat) (positive : 0 < sides)
    (sourceView : outcomes.View heap source (outcomes.all sides)) (table : Table heap tableRoot)
    (getSource : env "outcomes" = some (.ref source)) (getTable : env "landings" = some (.ref tableRoot))
    (getPos : env "pos" = some (number pos)) (getDoubles : env "doubles" = some (number doubles)) :
    ∃ (result : Entries) (extra : Mutable.Heap) (finalEnv : HeapCalls.Env),
      BlockContinues program env initialCode heap finalEnv (heap ++ extra) globals ∧ Found (heap ++ extra) heap.length result ∧
      finalEnv "row" = some (.ref heap.length) ∧ LocalsFrame env finalEnv ∧
      ∀ target, ((lookup result target).getD 0 : Int) =
        ((Monopoly.turnWeights sides (pos, doubles)).map fun entry =>
          if target = Monopoly.encode entry.1.1 entry.1.2 then (entry.2 : Int) else 0).sum := by
  obtain ⟨result, extra, finalEnv, ran, found, getRow, frame, counts⟩ :=
    build_result env heap globals source tableRoot pos doubles (outcomes.all sides) sourceView table
      getSource getTable getPos getDoubles (fun outcome member => by
        have bounds := outcomes.all_member sides positive outcome member
        exact ⟨by omega, Int.le_of_lt bounds.2.2⟩)
  exact ⟨result, extra, finalEnv, ran, found, getRow, frame,
    fun target => (counts target).trans (model_total sides pos doubles target positive)⟩

end solvers.«84».transition_row
