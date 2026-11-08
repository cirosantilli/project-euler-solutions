import solvers.«96.py.lean.d».RecordLoop

namespace solvers.«96».parse_puzzles
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

def normalizedHeap (text : String) (heap : Mutable.Heap) : Mutable.Heap :=
  heap ++ [.list ((Strings.splitLines text).map HeapTextLines.textValue),
    .list ((HeapTextLines.normalized text).map HeapTextLines.textValue)]

/-- The complete parser returns fresh row lists for exactly the records in the text contract. -/
theorem result (text : String) (solutions : List Board) (input : Sudoku.Text.Input text solutions)
    (heap : Mutable.Heap) (globals : Mutable.Globals) :
    ∃ boards, List.Forall₂ Correct boards solutions ∧
      ReturnsValue program "parse_puzzles" [.scalar (.string text)] heap (.ref (normalizedHeap text heap).length)
        (work (normalizedHeap text heap) boards) globals := by
  let entry := functionEnv definition.1 definition.2 [.scalar (.string text)] globals
  let withLines := setLocal entry "lines" (.ref (heap.length + 1))
  let withPuzzles := setLocal withLines "puzzles" (.ref (normalizedHeap text heap).length)
  let ready := setLocal withPuzzles "i" (number 0)
  have getText : entry "text" = some (.scalar (.string text)) := by cases globals <;> rfl
  have locals : Locals (heap.length + 1) (normalizedHeap text heap).length 0 ready := by
    constructor <;> simp [ready, withPuzzles, withLines, setLocal]
  obtain ⟨boards, correct, executed⟩ := loop_result input (HeapTextLines.normalized text) 0 (heap.length + 1)
    (normalizedHeap text heap) [] ready globals (by simp) (by simp [normalizedHeap]) locals
  refine ⟨boards, correct, ?_⟩
  apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
  rw [source]
  apply (normalize text heap entry globals getText).returns
  apply BlockReturns.assign (Evaluates.list .nil)
  apply BlockReturns.assign Evaluates.constant
  simpa only [List.nil_append] using executed

theorem «Returns» (text : String) (solutions : List Board) (input : Sudoku.Text.Input text solutions)
    (heap : Mutable.Heap) (globals : Mutable.Globals) :
    (HeapCalls.call program "parse_puzzles" [.scalar (.string text)] heap globals).Returns := by
  obtain ⟨_, _, executed⟩ := result text solutions input heap globals
  exact executed.returns

theorem «Supported» (text : String) (solutions : List Board) (input : Sudoku.Text.Input text solutions)
    (heap : Mutable.Heap) (globals : Mutable.Globals) :
    (HeapCalls.call program "parse_puzzles" [.scalar (.string text)] heap globals).Supported :=
  («Returns» text solutions input heap globals).supported

end
end solvers.«96».parse_puzzles
