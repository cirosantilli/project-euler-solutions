import solvers.«84.py.lean.d».LandingWeightUpdates

namespace solvers.«84».resolve_landing.weights
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

structure Card where
  row : Nat
  entry : Entry

def Card.View (card : Card) (heap : Mutable.Heap) : Prop :=
  heap[card.row]? = some (.tuple [.ref card.entry.address, .scalar (.int card.entry.weight)]) ∧
    TuplePool.View heap (.tuple (pool.pair card.entry.landing.1 card.entry.landing.2)) (.ref card.entry.address)

def bind (env : HeapCalls.Env) (card : Card) : HeapCalls.Env :=
  HeapCalls.assignNames env ["outcome", "count"] [.ref card.entry.address, .scalar (.int card.entry.weight)]

def applyCard (entries : List Entry) (card : Card) : List Entry :=
  add entries card.entry.landing card.entry.address card.entry.weight

theorem Card.View.frame {before after : Mutable.Heap} {card : Card} (view : card.View before)
    (kept : ∀ (address : Nat) (items : List Mutable.Value), before[address]? = some (.tuple items) → after[address]? = some (.tuple items)) :
    card.View after := ⟨kept card.row _ view.1, TuplePool.View.frame _ _ kept _ _ view.2⟩

/-- The unpacking loop visits each live row once and adds its count to the shared dictionary. -/
theorem card_loop (remaining before : List Card) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (source root : Nat) (entries : List Entry)
    (table : heap[source]? = some (.tuple ((before ++ remaining).map fun card => .ref card.row)))
    (views : ∀ card ∈ remaining, card.View heap)
    (found : heap[root]? = some (.keyDict (encode entries))) (getWeights : env "weights" = some (.ref root)) :
    HeapCalls.BlockContinues program env
      [.unpackStepLoop ["outcome", "count"] (.ref source) before.length [increment (.name "count")] []]
      heap (remaining.foldl bind env) (heap.set root (.keyDict (encode (remaining.foldl applyCard entries)))) globals := by
  induction remaining generalizing before env heap entries with
  | nil =>
      have unchanged : heap.set root (.keyDict (encode entries)) = heap := by
        obtain ⟨inside, same⟩ := List.getElem?_eq_some_iff.mp found
        rw [← same, List.set_getElem_self inside]
      simp only [List.foldl_nil, unchanged]
      apply HeapCalls.BlockContinues.unpack_done
        (items := before.map fun card => .ref card.row)
        (by simp [HeapCalls.sequenceValue, Mutable.sequence, table])
      · simp
      · exact HeapCalls.BlockContinues.nil
  | cons card rest ih =>
      have view := views card (by simp)
      let nextHeap := heap.set root (.keyDict (encode (applyCard entries card)))
      have kept := TupleDictionary.tuple_frame heap root (encode entries) (encode (applyCard entries card)) found
      have nextTable : nextHeap[source]? = some (.tuple (((before ++ [card]) ++ rest).map fun item => .ref item.row)) := by
        simpa [List.append_assoc] using kept source _ table
      have nextViews : ∀ item ∈ rest, item.View nextHeap :=
        fun item member => (views item (by simp [member])).frame kept
      have nextFound : nextHeap[root]? = some (.keyDict (encode (applyCard entries card))) :=
        List.getElem?_set_self (List.getElem?_eq_some_iff.mp found).1
      have getNextWeights : bind env card "weights" = some (.ref root) := by
        simpa [bind, HeapCalls.assignNames, HeapCalls.setLocal] using getWeights
      have run := increment_result (bind env card) heap globals root entries card.entry.landing card.entry.address card.entry.weight
        (.name "count") found getNextWeights (by simp [bind, HeapCalls.assignNames, HeapCalls.setLocal]) view.2
        (HeapCalls.Evaluates.lookup (by simp [bind, HeapCalls.assignNames, HeapCalls.setLocal]))
      have tail := ih (before ++ [card]) (bind env card) nextHeap (applyCard entries card) nextTable nextViews nextFound getNextWeights
      apply HeapCalls.BlockContinues.unpack_step
        (items := (before ++ card :: rest).map fun card => .ref card.row)
        (values := [.ref card.entry.address, .scalar (.int card.entry.weight)])
        (by simp [HeapCalls.sequenceValue, Mutable.sequence, table])
        (value := .ref card.row)
        (by simp [List.map_append])
        (by simp [Mutable.sequence, view.1]) rfl
      simpa [Calls.loopContinuation, Calls.loopControl, increment, List.foldl_cons, nextHeap, List.set_set] using run.append tail

end solvers.«84».resolve_landing.weights
