import solvers.«84.py.lean.d».TransitionItemsState

namespace solvers.«84».transition_row.items
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary resolve_landing
set_option Elab.async false

/-- One live dictionary item allocates its outer pair, unpacks the cached key, and updates the row. -/
theorem entry_result (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (root doubles count : Nat) (isDouble : Bool) (entry : weights.Entry) (entries : Entries)
    (found : Found heap root entries) (getRow : env "row" = some (.ref root))
    (getDoubles : env "doubles" = some (number doubles)) (getCount : env "count" = some (number count))
    (getDouble : env "is_double" = some (.scalar (.bool isDouble)))
    (key : TuplePool.View heap (.tuple (pool.pair entry.landing.1 entry.landing.2)) (.ref entry.address))
    (destinationNonneg : 0 ≤ entry.landing.1) (weightNonneg : 0 ≤ entry.weight) :
    ∃ finalHeap finalEnv,
      BlockContinues program (setLocal env "$formalsw.unpack" (.ref heap.length)) body (pairHeap heap entry) finalEnv finalHeap globals ∧
      Found finalHeap root (applyEntry doubles count isDouble entries entry) ∧
      HeapFrame heap finalHeap root ∧ LocalsFrame env finalEnv := by
  let bound := bindings.locals (setLocal env "$formalsw.unpack" (.ref heap.length)) entry
  let finalEnv := landingLocals bound entry.landing.1.toNat doubles entry.landing.2 isDouble
  let finalHeap := (pairHeap heap entry).set root (.dict (encode (applyEntry doubles count isDouble entries entry)))
  have extended := found_append found [.tuple [.ref entry.address, .scalar (.int entry.weight)]]
  have unpacked := bindings.result (pairHeap heap entry) (setLocal env "$formalsw.unpack" (.ref heap.length)) globals heap.length entry
    (by simp [setLocal]) (by simp [pairHeap]) (key_tuple _ entry (key.append _))
  have updated := landing_body_result bound (pairHeap heap entry) globals root entry.landing.1.toNat doubles entry.weight.toNat count
    entry.landing.2 isDouble entries extended
    (by simpa [bound, bindings.locals, assignNames, setLocal] using getRow)
    (by simp [bound, bindings.locals, assignNames, setLocal, Int.toNat_of_nonneg destinationNonneg])
    (by simpa [bound, bindings.locals, assignNames, setLocal] using getDoubles)
    (by simp [bound, bindings.locals, assignNames, setLocal, Int.toNat_of_nonneg weightNonneg])
    (by simpa [bound, bindings.locals, assignNames, setLocal] using getCount)
    (by simp [bound, bindings.locals, assignNames, setLocal])
    (by simpa [bound, bindings.locals, assignNames, setLocal] using getDouble)
  exact ⟨finalHeap, finalEnv, unpacked.append updated, found_set extended _, heap_frame _ _ _ _, locals_frame _ _ _ _ _⟩

end solvers.«84».transition_row.items
