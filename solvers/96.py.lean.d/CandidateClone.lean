import solvers.«96.py.lean.d».CandidateBits

namespace solvers.«96».solve_cands.clone
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

abbrev body : List Stmt :=
  [.assign [.name "nxt"] (.subscript (.name "cands") (.slice none none none)),
   .assign [.subscript (.name "nxt") (.name "idx")] (.name "b")]

def work (heap : Mutable.Heap) (candidates : Candidates) (cell : Cell) (digit : Digit) : Mutable.Heap :=
  heap ++ [.list (values (Function.update candidates cell {digit}))]

theorem source : (bits.body.drop 1).take 2 = body := rfl

/-- The copy and branch write allocate one independent candidate array. -/
theorem execute (address : Nat) (candidates : Candidates) (cell : Cell) (digit : Digit)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (found : View address heap candidates) (getCandidates : env "cands" = some (.ref address))
    (getIndex : env "idx" = some (number (Encoding.index cell).val))
    (getBit : env "b" = some (number (2 ^ digit.val))) :
    BlockContinues program env body heap (setLocal env "nxt" (.ref heap.length)) (work heap candidates cell digit) globals := by
  let copied := heap ++ [.list (values candidates)]
  let ready := setLocal env "nxt" (.ref heap.length)
  have allocated : Evaluates program env (.subscript (.name "cands") (.slice none none none)) heap
      (.ref heap.length) copied globals :=
    Evaluates.slice (.cons (Evaluates.lookup getCandidates) (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil))))
      (Mutable.getSlice_list_copy heap address (values candidates) found)
  have view : View heap.length copied candidates := by simp [View, copied]
  have index : ready "idx" = some (number (Encoding.index cell).val) := by simpa [ready, setLocal] using getIndex
  have bit : ready "b" = some (number (2 ^ digit.val)) := by simpa [ready, setLocal] using getBit
  apply (BlockContinues.assign allocated).append
  apply BlockContinues.setItem (receiver := .ref heap.length) (by intros; simp) (Evaluates.lookup bit)
    (.cons (Evaluates.lookup (by simp [ready, setLocal])) (.cons (Evaluates.lookup index) .nil))
  simpa [Masks.singleton, copied, write, work, List.set_append_right] using set_mask heap.length copied candidates cell {digit} view

theorem view (heap : Mutable.Heap) (candidates : Candidates) (cell : Cell) (digit : Digit) :
    View heap.length (work heap candidates cell digit) (Function.update candidates cell {digit}) := by simp [View, work]

theorem frame (heap : Mutable.Heap) (candidates : Candidates) (cell : Cell) (digit : Digit)
    (index : Nat) (inside : index < heap.length) : (work heap candidates cell digit)[index]? = heap[index]? :=
  List.getElem?_append_left inside

theorem nonempty (candidates : Candidates) (cell : Cell) (digit : Digit)
    (before : ∀ cell, candidates cell ≠ ∅) : ∀ other, Function.update candidates cell {digit} other ≠ ∅ := by
  intro other
  by_cases same : other = cell
  · subst other
    simp
  · simpa [Function.update_of_ne same] using before other

theorem branch (address : Nat) (candidates : Candidates) (cell : Cell) (digit : Digit)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (found : View address heap candidates) (getCandidates : env "cands" = some (.ref address))
    (getIndex : env "idx" = some (number (Encoding.index cell).val))
    (getBit : env "b" = some (number (2 ^ digit.val)))
    (multiple : 1 < (candidates cell).card) (allNonempty : ∀ cell, candidates cell ≠ ∅) :
    BlockContinues program env ((bits.body.drop 1).take 2) heap (setLocal env "nxt" (.ref heap.length))
      (work heap candidates cell digit) globals ∧
      View heap.length (work heap candidates cell digit) (Function.update candidates cell {digit}) ∧
      (∀ index < heap.length, (work heap candidates cell digit)[index]? = heap[index]?) ∧
      (∀ other, Function.update candidates cell {digit} other ≠ ∅) ∧
      weight (Function.update candidates cell {digit}) < weight candidates := by
  rw [source]
  exact ⟨execute address candidates cell digit env heap globals found getCandidates getIndex getBit,
    view heap candidates cell digit, frame heap candidates cell digit,
    nonempty candidates cell digit allNonempty, branch_weight_lt candidates cell digit multiple⟩

end
end solvers.«96».solve_cands.clone
