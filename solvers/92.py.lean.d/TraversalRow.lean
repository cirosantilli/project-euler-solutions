import solvers.«92.py.lean.d».TraversalState

namespace solvers.«92».traversal
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

/-- A yielded item shares the original tuple key and its current multiplicity. -/
def pairHeap (heap : Mutable.Heap) (entry : table.Entry) : Mutable.Heap :=
  heap ++ [.tuple [.ref entry.2.1, table.number entry.2.2]]

def pairLocals (heap : Mutable.Heap) (before : HeapCalls.Env) : HeapCalls.Env :=
  setLocal before "$formalsw.unpack" (.ref heap.length)

theorem allocated (heap : Mutable.Heap) (entry : table.Entry) :
    Mutable.allocateTuple [.ref entry.2.1, table.number entry.2.2] heap =
      some (.ok (.ref heap.length), pairHeap heap entry) := by
  simp [Mutable.allocateTuple, pairHeap, Except.map]

/-- Unpack a yielded item and execute every permitted digit for that row. -/
theorem row_result (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (destination ceiling : Nat) (entry : table.Entry) (accumulated : table.Entries)
    (following : before "following" = some (.ref destination))
    (limit : before "ceiling" = some (table.number ceiling))
    (key : heap[entry.2.1]? = some (table.tuple entry.1))
    (found : table.Found heap destination accumulated) (valid : table.Keys heap accumulated) :
    ∃ result finalHeap finalLocals,
      BlockContinues program (pairLocals heap before) body (pairHeap heap entry) finalLocals finalHeap globals ∧
      table.Found finalHeap destination result ∧ table.Keys finalHeap result ∧
      table.view result = Prefix.collect (table.view accumulated) (Prefix.row ceiling (entry.1, entry.2.2)) ∧
      rows.HeapFrame heap finalHeap destination ∧ LocalsFrame before finalLocals := by
  have pair : (pairHeap heap entry)[heap.length]? =
      some (.tuple [.ref entry.2.1, table.number entry.2.2]) := by simp [pairHeap]
  have saved : (pairHeap heap entry)[entry.2.1]? = some (table.tuple entry.1) :=
    (List.getElem?_append_left (List.getElem?_eq_some_iff.mp key).1).trans key
  have unpacked := bindings.result (pairHeap heap entry) (pairLocals heap before) globals
    heap.length entry.2.1 entry.2.2 entry.1 (by simp [pairLocals, setLocal]) pair saved
  obtain ⟨result, finalHeap, finalLocals, ran, finalFound, finalKeys, view, heapFrame, localFrame⟩ :=
    rows.result (pairHeap heap entry)
      (bindings.locals (pairLocals heap before) entry.2.1 entry.2.2 entry.1) globals
      destination entry.2.2 ceiling entry.1 accumulated
      (bindings.ready _ _ _ _ _ _
        (by simpa [pairLocals, setLocal] using following)
        (by simpa [pairLocals, setLocal] using limit))
      (found.append _) (valid.append _)
  refine ⟨result, finalHeap, finalLocals, unpacked.append ran, finalFound, finalKeys, view, ?_, ?_⟩
  · intro other object different present
    exact heapFrame other object different
      ((List.getElem?_append_left (List.getElem?_eq_some_iff.mp present).1).trans present)
  · exact locals_frame before finalLocals heap.length entry.2.1 entry.2.2 entry.1 localFrame

end solvers.«92».traversal
