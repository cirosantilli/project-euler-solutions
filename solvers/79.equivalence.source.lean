import solvers.«79.outcome.source»
import lang.py.FormalSW.MutableRelocation

/-! Source-level equivalence compares returned values, exceptions, and all caller-owned objects. -/
namespace solvers.«79».source
open FormalSW.PyAst HeapCalls
open solvers.«79».graph solvers.«79».topo
open ProjectEulerStatements.TopologicalOrder
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

theorem solve_outcome_within (attempts : List String) (anchors : Nat) (value : Mutable.Value)
    (returned : solveOutcome attempts = .ok value) : value.edge.Within anchors := by
  unfold solveOutcome at returned
  cases processed : emptyStorage.graphData.process attempts with
  | error error => simp only [processed] at returned; contradiction
  | ok data =>
      simp only [processed] at returned
      cases ordered : order data.vertices data.edges with
      | none => simp only [ordered, resultOutcome] at returned; contradiction
      | some output =>
          simp only [ordered, resultOutcome, Except.ok.injEq] at returned
          rw [← returned]
          trivial

theorem outcome_allocations (base left right : Mutable.Heap) (valid : base.Valid)
    (leftSaved : base <+: left) (rightSaved : base <+: right)
    (attempts : List String) :
    Mutable.Completion.Equivalent ⟨solveOutcome attempts, [], left, base.length⟩
      ⟨solveOutcome attempts, [], right, base.length⟩ := by
  obtain ⟨leftExtra, rfl⟩ := leftSaved
  obtain ⟨rightExtra, rfl⟩ := rightSaved
  cases outcome : solveOutcome attempts with
  | ok value =>
      exact Mutable.Completion.equivalent_allocations base leftExtra rightExtra value [] valid
        (solve_outcome_within attempts base.length value outcome) (by simp)
  | error error =>
      exact Mutable.Relocation.equivalent_exception_allocations base leftExtra rightExtra error []
        valid (by simp)

/-- Every optimized traversal order agrees with the reference on each stable sequence input. -/
theorem sequence_equiv (choices : ChoiceSchedule) (base : Mutable.Heap) (value : Mutable.Value)
    (attempts : List String) (valid : base.Valid)
    (input : SequenceInput base (sequenceSource value) attempts) :
    ((SetCalls.call program "solve" [value] base).run choices).Equiv
      (call p79.program "naive" [value] base) := by
  obtain ⟨left, optimized, leftSaved⟩ := solve_sequence_finishes choices base value attempts input
  obtain ⟨right, reference, rightSaved⟩ := p79.naive_sequence_finishes base value attempts
    ⟨input.read⟩
  rw [← solve_outcome_eq_reference] at reference
  exact Mutable.Call.equiv_of_finishes optimized reference
    (outcome_allocations base left right valid leftSaved rightSaved attempts)

theorem list_equiv (choices : ChoiceSchedule) (base : Mutable.Heap) (address : Nat)
    (attempts : List String) (valid : base.Valid)
    (found : base[address]? = some (.list (attempts.map text))) :
    ((SetCalls.call program "solve" [.ref address] base).run choices).Equiv
      (call p79.program "naive" [.ref address] base) :=
  sequence_equiv choices base (.ref address) attempts valid
    (SequenceInput.list base address attempts found)

theorem tuple_equiv (choices : ChoiceSchedule) (base : Mutable.Heap) (address : Nat)
    (attempts : List String) (valid : base.Valid)
    (found : base[address]? = some (.tuple (attempts.map text))) :
    ((SetCalls.call program "solve" [.ref address] base).run choices).Equiv
      (call p79.program "naive" [.ref address] base) :=
  sequence_equiv choices base (.ref address) attempts valid
    (SequenceInput.tuple base address attempts found)

theorem sequence_family_equiv (base : Mutable.Heap) (value : Mutable.Value)
    (attempts : List String) (valid : base.Valid)
    (input : SequenceInput base (sequenceSource value) attempts) :
    (SetCalls.call program "solve" [value] base).Equiv
      (Mutable.CallFamily.deterministic (call p79.program "naive" [value] base)) :=
  fun choices _ => sequence_equiv choices base value attempts valid input

/-- A string is itself iterable; its nonempty inputs fail validation at the first character. -/
theorem string_equiv (choices : ChoiceSchedule) (base : Mutable.Heap) (value : String)
    (valid : base.Valid) :
    ((SetCalls.call program "solve" [text value] base).run choices).Equiv
      (call p79.program "naive" [text value] base) := by
  apply sequence_equiv choices base (text value) (value.toList.map String.singleton) valid
  constructor
  intro heap _
  simp [sequenceSource, sequenceValue, Mutable.sequence, List.map_map, text]

end solvers.«79».source
