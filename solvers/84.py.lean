import solvers.«84.py.lean.d».StationarySetup

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
