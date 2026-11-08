import solvers.«96.py.lean.d».CandidatePeerTable

namespace solvers.«96».reduce_cands.queueLoop
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

abbrev statement : Stmt := .whileDo (.name "queue") queueBody []

theorem source : outerBody[2]? = some statement := rfl
theorem source_body : queueBody = queue.takeLast ++ [peerLoop.statement] := rfl

structure Locals (table : PeerTable) (address queue : Nat) (state : Propagation.State) (env : HeapCalls.Env) : Prop where
  candidates : env "cands" = some (.ref address)
  queue : env "queue" = some (.ref queue)
  changed : env "changed" = some (.scalar (.bool state.changed))
  peers : env "PEERS" = some (.ref table.root)

def ready (env : HeapCalls.Env) (cell : Cell) (digit : Digit) : HeapCalls.Env :=
  setLocal (setLocal env "i" (number (Encoding.index cell).val)) "val" (number (2 ^ digit.val))

def popped (heap : Mutable.Heap) (queue : Nat) (remaining : List Cell) : Mutable.Heap :=
  heap.set queue (.list (peerLoop.encoded remaining))

theorem pop_result (table : PeerTable) (address queue : Nat) (state : Propagation.State) (cell : Cell) (digit : Digit)
    (remaining : List Cell) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (view : peerLoop.View address queue heap state) (locals : Locals table address queue state env)
    (different : address ≠ queue) (queued : state.queued = remaining ++ [cell]) (single : state.candidates cell = {digit}) :
    BlockContinues program env reduce_cands.queue.takeLast heap (ready env cell digit) (popped heap queue remaining) globals := by
  have found : heap[queue]? = some (.list (peerLoop.encoded remaining ++ [number (Encoding.index cell).val])) := by
    simpa only [queued, peerLoop.encoded, List.map_append, List.map_cons, List.map_nil] using view.queued
  simpa only [Encoding.cell_index, single, Masks.singleton] using reduce_cands.queue.take_result address queue heap state.candidates
    (Encoding.index cell) (peerLoop.encoded remaining) env globals view.candidates found different locals.candidates locals.queue

theorem pop_frame (address queue : Nat) (heap : Mutable.Heap) (remaining : List Cell) :
    peerLoop.Frame address queue heap (popped heap queue remaining) :=
  ⟨List.length_set, fun _ _ notQueue => List.getElem?_set_ne notQueue⟩

theorem pop_view (address queue : Nat) (heap : Mutable.Heap) (state : Propagation.State) (remaining : List Cell)
    (view : peerLoop.View address queue heap state) (different : address ≠ queue) :
    peerLoop.View address queue (popped heap queue remaining) (Propagation.dequeued state remaining) :=
  ⟨(List.getElem?_set_ne different.symm).trans view.candidates,
    List.getElem?_set_self (List.getElem?_eq_some_iff.mp view.queued).1⟩

theorem ready_locals (table : PeerTable) (address queue : Nat) (state : Propagation.State) (cell : Cell) (digit : Digit)
    (remaining : List Cell) (env : HeapCalls.Env) (locals : Locals table address queue state env) :
    peerLoop.Locals address queue digit (Propagation.dequeued state remaining) (ready env cell digit) := by
  constructor
  · simpa [ready, setLocal] using locals.candidates
  · simpa [ready, setLocal] using locals.queue
  · simp [ready, setLocal]
  · simpa [ready, setLocal] using locals.changed

theorem enter (choices : ChoiceSchedule) (globals : Mutable.Globals) (table : PeerTable) (address queue : Nat)
    (state : Propagation.State) (cell : Cell) (remaining : List Cell) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (view : peerLoop.View address queue heap state) (locals : Locals table address queue state env)
    (queued : state.queued = remaining ++ [cell])
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    SetCalls.Reaches choices program anchors
      ⟨⟨.block (statement :: following) env, stack, heap⟩, used⟩
      ⟨⟨.block (queueBody ++ statement :: following) env, stack, heap⟩, used⟩ globals := by
  apply SetCalls.Reaches.of_heap
  apply HeapCalls.Reaches.next (following := ⟨.expr (.name "queue") env,
    .branch (queueBody ++ [Calls.loopContinuation queueBody statement]) [] following env :: stack, heap⟩) rfl
  apply (Evaluates.lookup (globals := globals) (module := program) (heap := heap) locals.queue anchors _).trans
  have truth : Mutable.truth (.ref queue) heap = some (.ok true, heap) := by
    simp [Mutable.truth, view.queued, queued, peerLoop.encoded]
  exact .next (by simp [step, run, truth, Calls.loopContinuation, show Calls.loopControl queueBody = false from rfl]) .refl

theorem finish (choices : ChoiceSchedule) (globals : Mutable.Globals) (table : PeerTable) (address queue : Nat)
    (state : Propagation.State) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (view : peerLoop.View address queue heap state) (locals : Locals table address queue state env)
    (empty : state.queued = [])
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    SetCalls.Reaches choices program anchors
      ⟨⟨.block (statement :: following) env, stack, heap⟩, used⟩
      ⟨⟨.block following env, stack, heap⟩, used⟩ globals := by
  apply SetCalls.Reaches.of_heap
  apply HeapCalls.Reaches.next (following := ⟨.expr (.name "queue") env,
    .branch (queueBody ++ [Calls.loopContinuation queueBody statement]) [] following env :: stack, heap⟩) rfl
  apply (Evaluates.lookup (globals := globals) (module := program) (heap := heap) locals.queue anchors _).trans
  have truth : Mutable.truth (.ref queue) heap = some (.ok false, heap) := by
    simp [Mutable.truth, view.queued, empty, peerLoop.encoded]
  exact .next (by simp [step, run, truth]) .refl

end
end solvers.«96».reduce_cands.queueLoop
