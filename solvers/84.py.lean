import solvers.«84.py.lean.d».SolveResult
import data.«project-euler-statements».data.py.project_euler_statements.«p84.py»

namespace solvers.«84».encode_state
open FormalSW.PyAst HeapCalls

theorem «Returns» (square doubles : Int) :
    (HeapCalls.call program "encode_state"
      [.scalar (.int square), .scalar (.int doubles)] []).Returns :=
  (result square doubles []).returns

theorem «Supported» (square doubles : Int) :
    (HeapCalls.call program "encode_state"
      [.scalar (.int square), .scalar (.int doubles)] []).Supported :=
  («Returns» square doubles).supported

end solvers.«84».encode_state

namespace solvers.«84».resolve_landing
open FormalSW.PyAst ProjectEulerSolutions

theorem «Returns» (square : Int) :
    (HeapMemoCalls.call program "resolve_landing" [.scalar (.int square)] []).Returns := by
  by_cases jail : square = 30
  · subst square
    exact (jail_result [] []).returns
  · by_cases cc : square ∈ [2, 17, 33]
    · exact (community.result square cc [] []).returns
    · by_cases ch : square ∈ [7, 22, 36]
      · exact (chance.result square ch [] []).returns
      · apply (HeapMemoCalls.ReturnsValue.from_heap (spec := pool.spec) (HeapCalls.Extends.refl program) pool.disjoint
          (ordinary_result square [] [] jail cc ch) ?_).rel.returns
        simpa only [List.nil_append] using TuplePool.preserves_append pool.catalog []
          (singletonHeap [.tuple [.scalar (.int 2), .scalar (.int 17), .scalar (.int 33)],
            .tuple [.scalar (.int 7), .scalar (.int 22), .scalar (.int 36)]] square false)

theorem «Supported» (square : Int) :
    (HeapMemoCalls.call program "resolve_landing" [.scalar (.int square)] []).Supported :=
  («Returns» square).supported

end solvers.«84».resolve_landing

namespace solvers.«84».build_transition
open FormalSW.PyAst

theorem «Returns» (sides : Nat) (positive : 0 < sides) :
    (HeapMemoCalls.call program "build_transition" [.scalar (.int sides)] []).Returns :=
  (result sides positive [] []).returns

theorem «Supported» (sides : Nat) (positive : 0 < sides) :
    (HeapMemoCalls.call program "build_transition" [.scalar (.int sides)] []).Supported :=
  («Returns» sides positive).supported

end solvers.«84».build_transition

namespace solvers.«84».solve
open FormalSW.PyAst

theorem «Returns» (sides : Nat) (positive : 0 < sides) (heap : Mutable.Heap := [])
    (globals : Mutable.Globals := []) :
    (HeapMemoCalls.call program "solve" [.scalar (.int sides)] heap globals).Returns :=
  (result sides positive heap globals).returns

theorem «Supported» (sides : Nat) (positive : 0 < sides) (heap : Mutable.Heap := [])
    (globals : Mutable.Globals := []) :
    (HeapMemoCalls.call program "solve" [.scalar (.int sides)] heap globals).Supported :=
  («Returns» sides positive heap globals).supported

/-- Both complete source calls return the same modal string and preserve caller objects and aliases. -/
theorem «Equiv(naive)» (sides : Nat) (positive : 0 < sides) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (valid : heap.Valid)
    (globalsWithin : ∀ entry ∈ globals, entry.2.edge.Within heap.length) :
    (HeapMemoCalls.call program "solve" [.scalar (.int sides)] heap globals).Equiv
      (HeapMemoCalls.call P84Ast.module "naive" [.scalar (.int sides)] heap globals) := by
  obtain ⟨_, leftHeap, ⟨leftWeights, leftValid, rfl, leftFrame⟩, leftFinished⟩ :=
    (result sides positive heap globals).finishes
  obtain ⟨_, rightHeap, ⟨rightWeights, rightValid, rfl, rightFrame⟩, rightFinished⟩ :=
    (data.«project-euler-statements».data.py.project_euler_statements.p84.naive.result
      sides positive heap globals).finishes
  have leftPrefix : heap <+: leftHeap := List.prefix_iff_getElem?.mpr
    (fun index inside => (leftFrame index inside).trans (List.getElem?_eq_getElem inside))
  have rightPrefix : heap <+: rightHeap := List.prefix_iff_getElem?.mpr
    (fun index inside => (rightFrame index inside).trans (List.getElem?_eq_getElem inside))
  obtain ⟨leftExtra, rfl⟩ := leftPrefix
  obtain ⟨rightExtra, rfl⟩ := rightPrefix
  rw [ProjectEulerSolutions.MonopolyOutput.answer_equal positive leftValid rightValid] at leftFinished
  exact Mutable.Call.equiv_of_finishes leftFinished rightFinished
    (Mutable.Completion.equivalent_allocations heap leftExtra rightExtra
      (.scalar (.string (ProjectEulerSolutions.MonopolyOutput.answer rightWeights))) globals
      valid trivial globalsWithin)

end solvers.«84».solve
