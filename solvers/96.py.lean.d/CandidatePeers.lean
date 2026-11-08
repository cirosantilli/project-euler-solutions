import solvers.«96.py.lean.d».CandidateCells

namespace solvers.«96».reduce_cands.peer
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

abbrev emptyTest : Expr := .compare candidate [(.eq, .constant (.int 0))]
abbrev singleTest : Expr := .compare (.builtinCall .intBitCount [candidate]) [(.eq, .constant (.int 1))]
abbrev reject : Stmt := .ifThen emptyTest [.return (.constant (.bool false))] []
abbrev enqueue : Stmt := .ifThen singleTest [.expr (.call (.attribute (.name "queue") "append") [.name "p"])] []
abbrev changed : Stmt := .assign [.name "changed"] (.constant (.bool true))

theorem source : peerBody = [.ifThen test [clear, reject, enqueue, changed] []] := rfl

structure Locals (env : HeapCalls.Env) (address queue : Nat) (position : Cell) (digit : Digit) : Prop where
  candidates : env "cands" = some (.ref address)
  position : env "p" = some (number (Encoding.index position).val)
  digit : env "val" = some (number (2 ^ digit.val))
  queue : env "queue" = some (.ref queue)

theorem read (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address))
    (getPosition : env "p" = some (number (Encoding.index position).val)) :
    Evaluates program env candidate heap (number (Masks.encode (candidates position))) heap globals := by
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getCandidates) (Evaluates.lookup getPosition)
  simpa only [Encoding.cell_index] using get_mask address heap candidates (Encoding.index position) found

theorem empty_test (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address))
    (getPosition : env "p" = some (number (Encoding.index position).val)) :
    Evaluates program env emptyTest heap (.scalar (.bool (decide (candidates position = ∅)))) heap globals := by
  apply Evaluates.compare (middle := heap) (afterRight := heap)
    (read address heap candidates position env globals found getCandidates getPosition) Evaluates.constant
  simp [HeapCalls.compare, scalarCompare, evalCmpOp, Masks.zero_iff]

theorem single_test (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address))
    (getPosition : env "p" = some (number (Encoding.index position).val)) :
    Evaluates program env singleTest heap (.scalar (.bool (decide ((candidates position).card = 1)))) heap globals := by
  apply Evaluates.compare (middle := heap) (afterRight := heap)
    (av := number (candidates position).card) (bv := number 1)
  · apply Evaluates.builtin (.cons (read address heap candidates position env globals found getCandidates getPosition) .nil)
    change (some (.ok (number (FormalSW.Py.Builtin.Int.bitCount (Masks.encode (candidates position)))), heap) :
      Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
    rw [Masks.bitCount]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    omega

theorem skip (address queue : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell) (digit : Digit)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : View address heap candidates)
    (locals : Locals env address queue position digit) (absent : digit ∉ candidates position) :
    BlockContinues program env peerBody heap env heap globals := by
  rw [source]
  apply BlockContinues.branch (flag := false)
    (test_result address heap candidates position digit env globals found locals.candidates locals.position locals.digit)
  · simpa only [decide_eq_false absent] using test_truth heap (candidates position) digit
  · exact .nil

theorem enqueue_result (address queue : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell) (digit : Digit)
    (items : List Mutable.Value) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : View address heap candidates) (queued : heap[queue]? = some (.list items))
    (locals : Locals env address queue position digit) :
    BlockContinues program env [enqueue] heap env
      (if (candidates position).card = 1 then heap.set queue (.list (items ++ [number (Encoding.index position).val])) else heap) globals := by
  have tested := single_test address heap candidates position env globals found locals.candidates locals.position
  by_cases single : (candidates position).card = 1
  · rw [if_pos single]
    apply BlockContinues.branch (flag := true) tested (by simp [Mutable.truth, scalarTruth, single])
    exact BlockContinues.discard (Evaluates.append (Evaluates.lookup locals.queue) queued (Evaluates.lookup locals.position) queued)
  · rw [if_neg single]
    exact BlockContinues.branch (flag := false) tested (by simp [Mutable.truth, scalarTruth, single]) .nil

def finish (address queue : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell) (digit : Digit)
    (items : List Mutable.Value) : Mutable.Heap :=
  let after := write address heap candidates position ((candidates position).erase digit)
  if ((candidates position).erase digit).card = 1 then
    after.set queue (.list (items ++ [number (Encoding.index position).val]))
  else after

theorem changed_result (address queue : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell) (digit : Digit)
    (items : List Mutable.Value) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : View address heap candidates) (queued : heap[queue]? = some (.list items))
    (different : address ≠ queue) (locals : Locals env address queue position digit)
    (present : digit ∈ candidates position) (nonempty : (candidates position).erase digit ≠ ∅) :
    BlockContinues program env peerBody heap (setLocal env "changed" (.scalar (.bool true)))
      (finish address queue heap candidates position digit items) globals := by
  let after := Function.update candidates position ((candidates position).erase digit)
  let work := write address heap candidates position ((candidates position).erase digit)
  have viewed : View address work after := write_view address heap candidates position _ found
  have queueKept : work[queue]? = some (.list items) :=
    (write_frame address queue heap candidates position _ different).trans queued
  have skipped : BlockContinues program env [reject] work env work globals := by
    apply BlockContinues.branch (flag := false)
      (empty_test address work after position env globals viewed locals.candidates locals.position)
    · simp [Mutable.truth, scalarTruth, after, nonempty]
    · exact .nil
  have enqueueDone := enqueue_result address queue work after position digit items env globals viewed queueKept locals
  have queueResult : BlockContinues program env [enqueue] work env
      (finish address queue heap candidates position digit items) globals := by
    simpa only [after, Function.update_self] using enqueueDone
  rw [source]
  apply BlockContinues.branch (flag := true)
    (test_result address heap candidates position digit env globals found locals.candidates locals.position locals.digit)
  · simpa only [decide_eq_true present] using test_truth heap (candidates position) digit
  · exact (clear_result address heap candidates position digit env globals found locals.candidates locals.position locals.digit).append
      (skipped.append (queueResult.append (BlockContinues.assign Evaluates.constant)))

theorem finish_view (address queue : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell) (digit : Digit)
    (items : List Mutable.Value) (found : View address heap candidates) (different : address ≠ queue) :
    View address (finish address queue heap candidates position digit items)
      (Function.update candidates position ((candidates position).erase digit)) := by
  unfold finish View
  split
  · rw [List.getElem?_set_ne different.symm]
    exact write_view address heap candidates position _ found
  · exact write_view address heap candidates position _ found

theorem finish_queue (address queue : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell) (digit : Digit)
    (items : List Mutable.Value) (queued : heap[queue]? = some (.list items)) (different : address ≠ queue) :
    (finish address queue heap candidates position digit items)[queue]? =
      some (.list (items ++ if ((candidates position).erase digit).card = 1 then [number (Encoding.index position).val] else [])) := by
  have kept := (write_frame address queue heap candidates position ((candidates position).erase digit) different).trans queued
  unfold finish
  split
  · exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp kept).1
  · simpa using kept

theorem finish_frame (address queue outside : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell) (digit : Digit)
    (items : List Mutable.Value) (notCandidates : address ≠ outside) (notQueue : queue ≠ outside) :
    (finish address queue heap candidates position digit items)[outside]? = heap[outside]? := by
  unfold finish
  split
  · rw [List.getElem?_set_ne notQueue]
    exact write_frame address outside heap candidates position _ notCandidates
  · exact write_frame address outside heap candidates position _ notCandidates

theorem conflict_result (address queue : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell) (digit : Digit)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : View address heap candidates)
    (locals : Locals env address queue position digit) (present : digit ∈ candidates position)
    (empty : (candidates position).erase digit = ∅) (following : List Stmt) :
    BlockReturns program env (peerBody ++ following) heap (.scalar (.bool false))
      (write address heap candidates position ((candidates position).erase digit)) globals := by
  let after := Function.update candidates position ((candidates position).erase digit)
  let work := write address heap candidates position ((candidates position).erase digit)
  have viewed : View address work after := write_view address heap candidates position _ found
  rw [source]
  apply BlockReturns.branch (flag := true)
    (test_result address heap candidates position digit env globals found locals.candidates locals.position locals.digit)
  · simpa only [decide_eq_true present] using test_truth heap (candidates position) digit
  · apply (clear_result address heap candidates position digit env globals found locals.candidates locals.position locals.digit).returns
    apply BlockReturns.branch (flag := true)
      (empty_test address work after position env globals viewed locals.candidates locals.position)
    · simp [Mutable.truth, scalarTruth, after, empty]
    · exact BlockReturns.return_value Evaluates.constant

/-- A contradiction reached by eliminating a singleton from its peer rules out every legal completion. -/
theorem conflict_impossible (candidates : Candidates) (cell other : Cell) (digit : Digit)
    (single : candidates cell = {digit}) (peer : Peer cell other) (empty : (candidates other).erase digit = ∅) :
    ¬ ∃ assignment, Legal assignment ∧ Admits candidates assignment := by
  rintro ⟨assignment, legal, admitted⟩
  have after := (erase_peer_iff candidates assignment legal single peer).mpr admitted
  exact empty_cell_impossible (Function.update candidates other ((candidates other).erase digit)) (cell := other)
    (by simpa using empty) ⟨assignment, after⟩

end
end solvers.«96».reduce_cands.peer
