import solvers.«96.py.lean.d».GridLoops

namespace solvers.«96».grid_to_cands
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false

/-- Both actual range loops initialize all 81 masks, preserving every object in the caller's heap. -/
theorem result (board : Board) (heap : Mutable.Heap) (address : Nat) (globals : Mutable.Globals)
    (grid : heap[address]? = some (.list ((Encoding.rows board).map text)))
    (allDigits : globals.lookup "ALL" = some (number 511)) :
    ReturnsValue program "grid_to_cands" [.ref address] heap (.ref (heap.length + 1))
      (work heap board 81) globals := by
  let entry := functionEnv definition.1 definition.2 [.ref address] globals
  let ready := setLocal entry "cands" (.ref (heap.length + 1))
  have getAll : entry "ALL" = some (number 511) := by
    cases globals with
    | nil => simp at allDigits
    | cons first rest =>
        have notLocal : "ALL" ∉ blockLocals definition.2 := by
          rw [source]
          simp [blockLocals, assignedLocals, targetLocals]
        simp only [entry, functionEnv, show "ALL" ∉ definition.1.map Arg.name from by decide,
          notLocal, if_false]
        exact allDigits
  have getGrid : entry "grid_lines" = some (.ref address) := by cases globals <;> rfl
  have allocation : Evaluates program entry
      (.binOp (.list [.name "ALL"]) .mult (.constant (.int 81))) heap
      (.ref (heap.length + 1)) (work heap board 0) globals := by
    apply Evaluates.binary (av := .ref heap.length) (bv := number 81)
      (middle := heap ++ [.list [number 511]]) (afterRight := heap ++ [.list [number 511]])
    · exact Evaluates.list (.cons (Evaluates.lookup getAll) .nil)
    · exact Evaluates.constant
    · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
        Mutable.array, Mutable.allocate, work, Encoding.prepared_zero, Except.map]
  obtain ⟨finalEnv, executed, kept⟩ := rows heap board 0 9 address rfl ready globals grid
    (by simpa [ready, setLocal] using getGrid) (by simp [ready, setLocal])
  have getCandidates : finalEnv "cands" = some (.ref (heap.length + 1)) := by
    rw [kept "cands" (by decide)]
    simp [ready, setLocal]
  apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
  rw [source]
  apply BlockReturns.assign allocation
  apply BlockReturns.for_range (.cons Evaluates.constant .nil) rfl
  exact executed.returns (BlockReturns.return_value (Evaluates.lookup getCandidates))

theorem candidate (board : Board) (heap : Mutable.Heap) (position : Cell) :
    Mutable.getItem (.ref (heap.length + 1)) (number (Encoding.index position).val) (work heap board 81) =
      some (.ok (number (Masks.encode (initialCandidates board position))), work heap board 81) := by
  let slot := Encoding.index position
  have selected := Encoding.prepared_get board 81 slot
  simp only [if_pos slot.isLt, slot, Encoding.cell_index, Encoding.mask] at selected
  obtain ⟨_, selected⟩ := List.getElem_of_getElem? selected
  have nonnegative : ¬ ((Encoding.index position).val : Int) < 0 := by omega
  simp [Mutable.getItem, Mutable.index, not_le.mpr (Encoding.index position).isLt,
    nonnegative, selected, Except.map]

theorem «Returns» (board : Board) (heap : Mutable.Heap) (address : Nat) (globals : Mutable.Globals)
    (grid : heap[address]? = some (.list ((Encoding.rows board).map text)))
    (allDigits : globals.lookup "ALL" = some (number 511)) :
    (HeapCalls.call program "grid_to_cands" [.ref address] heap globals).Returns :=
  (result board heap address globals grid allDigits).returns

theorem «Supported» (board : Board) (heap : Mutable.Heap) (address : Nat) (globals : Mutable.Globals)
    (grid : heap[address]? = some (.list ((Encoding.rows board).map text)))
    (allDigits : globals.lookup "ALL" = some (number 511)) :
    (HeapCalls.call program "grid_to_cands" [.ref address] heap globals).Supported :=
  («Returns» board heap address globals grid allDigits).supported

end solvers.«96».grid_to_cands
