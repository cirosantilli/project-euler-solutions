import solvers.«84.py.lean.d».LandingJail
import solvers.«84.py.lean.d».LandingResult

namespace solvers.«84».resolve_landing
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

theorem singleton_post (square : Int) (jailed : Bool) (weight : Int) (positive : 0 < weight)
    (heap : Mutable.Heap) (value : Mutable.Value) (after : Mutable.Heap)
    (post : SingletonPost heap square jailed weight value after) :
    weights.Post heap [((square, jailed), weight)] value after := by
  obtain ⟨address, root, extra, returned, same, fresh, found, key⟩ := post
  refine ⟨root, [⟨(square, jailed), address, weight⟩], extra, returned, same, fresh, ?_, ?_, ?_, ?_, ?_⟩
  · simpa [weights.encode] using found
  · intro entry member
    have only : entry = ⟨(square, jailed), address, weight⟩ := by simpa using member
    subst entry
    exact (TuplePool.view_scalars after [.int square, .bool jailed] _).mpr ⟨address, rfl, key⟩
  · simp [weights.Distinct]
  · simpa [weights.Positive] using positive
  · intro landing
    by_cases equal : landing = (square, jailed) <;> simp [weights.lookup, weights.total, equal, eq_comm]

theorem jail_weights_result (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.ReturnsRel program pool.spec "resolve_landing" [.scalar (.int 30)] heap
      (weights.Post heap [((10, true), 256)]) globals :=
  (jail_result heap globals).consequence (singleton_post 10 true 256 (by decide) heap)

theorem ordinary_weights_result (square : Int) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (notJail : square ≠ 30) (notCommunity : square ∉ [2, 17, 33]) (notChance : square ∉ [7, 22, 36]) :
    HeapMemoCalls.ReturnsRel program pool.spec "resolve_landing" [.scalar (.int square)] heap
      (weights.Post heap [((square, false), 256)]) globals := by
  let tests := [Mutable.Object.tuple [.scalar (.int 2), .scalar (.int 17), .scalar (.int 33)],
    .tuple [.scalar (.int 7), .scalar (.int 22), .scalar (.int 36)]]
  have run := HeapMemoCalls.ReturnsValue.from_heap (spec := pool.spec) (HeapCalls.Extends.refl program) pool.disjoint
    (ordinary_result square heap globals notJail notCommunity notChance) (by
      intro cache valid
      apply TuplePool.preserves_append pool.catalog (heap ++ tests) _ cache
      exact TuplePool.preserves_append pool.catalog heap tests cache valid)
  apply run.rel.consequence
  rintro value after ⟨rfl, same⟩
  subst after
  apply singleton_post square false 256 (by decide) heap
  refine ⟨heap.length + 2, heap.length + 3,
    tests ++ [.tuple [.scalar (.int square), .scalar (.bool false)],
      .keyDict [(.tuple (pool.pair square false), .ref (heap.length + 2), .scalar (.int 256))]], rfl, ?_, by omega, ?_, ?_⟩
  · simp [singletonHeap, tests, pool.pair, List.append_assoc]
  · simp [singletonHeap, pool.pair]
  · simp [singletonHeap]

end solvers.«84».resolve_landing
